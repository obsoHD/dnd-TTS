"""The Writer: one resident model, one pass, never in the render path.

Contract: docs/M3-contracts.md, section ``app/llm.py``. Why it is shaped like
this (REBUILD.md §3): the previous build let a "director" decide per line how
the character should sound, and the character stopped sounding like the same
person. So the model here touches **text only**. It never picks a delivery, it
never sees a take, and nothing it returns reaches the TTS without going through
``app.canon`` first, exactly like a line the DM typed by hand.

``fix`` does two jobs in one call because the DM types fast at the table with a
keyboard that fights Slovak diacritics: it corrects the sentence and it rewrites
it so a character can speak it out loud. Everything the prompt asks for is
re-checked in code afterwards (``_guard``): a mid-size model asked nicely still
emits control tokens, stage directions and three-times-longer paragraphs, and a
line that fails a guard is not worth arguing with - the DM gets their own text
back with a note and taps Speak.
"""
from __future__ import annotations

import json
import logging
import re
import threading

import requests

from app import canon, config
from app.voices import Voice

log = logging.getLogger(__name__)

RESIDENT_FRACTION = 0.9     # below this ollama has spilled the model to CPU: too slow for the table
PROBE_TIMEOUT_S = 3.0       # WHY shorter than the chat call: the pencil must go grey fast, not hang
CHAT_TIMEOUT_S = 8.0
WARM_TIMEOUT_S = 300.0      # a cold 27B takes up to a minute and a half off a spinning disk
_warming = threading.Event()
TEMPERATURE = 0.3           # a corrector, not an author: the same line twice should come back the same
NUM_CTX = 4096              # one line plus a persona; a 40k context would waste ~8 GB of VRAM
MAX_BEATS = 3               # canon caps pause tokens at 3 too, so a fourth beat would be dropped anyway
ATTEMPTS = 2                # a rejected answer is worth one more roll; a usable one is not
MIN_RATIO = 0.5
MAX_RATIO = 2.0

_TOKEN = re.compile(r"<\|[^|<>]*\|>")
_THINK = re.compile(r"<think>.*?</think>", re.S)
# The three ways a beat can arrive: em dash, ellipsis (character or dots), spaced
# hyphen. Same set canon reads, so what is capped here is what canon would hear.
_BEAT = re.compile(r"\s*—\s*|\s*(?:…|\.{3,})\s*|\s+-\s+")
_SPACED_BEATS = ("—", "-")
_QUOTES = "\"'„“”«»‚‘’"


class BrainNotReady(RuntimeError):
    """The 27B is not resident. There is no fallback model on purpose (§3): a
    smaller brain writes a different character, and the DM would not be told."""


# Both prompts are monolingual. A Slovak system prompt plus "answer in English"
# makes a mid-size model blend the two languages inside one sentence.
SK_FIX_SYSTEM = (
    "Si korektor a dialógový redaktor pre slovenský stôl Dungeons & Dragons. "
    "Dostaneš jednu repliku, ktorú postava o chvíľu povie nahlas.\n"
    # 0: the failure this prompt exists to prevent. A mid-size model reads a
    # question and answers it; the pencil must hand back the DM's own line.
    "0. NEODPOVEDAJ na repliku. Nie si postava a nevedieš rozhovor. Dostaneš text "
    "a vrátiš TEN ISTÝ text opravený. Ak je replika otázka, vrátiš tú istú otázku.\n"
    # 1: the DM types at the table, often without diacritics; this is the half of
    # the job that makes "ludia" into "ľudia" and keeps the case endings honest.
    "1. Oprav pravopis, diakritiku, pády, zhodu a slovosled. Odstráň bohemizmy "
    "(vždyť, doporučiť, tady, prostě) a anglické kalky.\n"
    # 2: written word order is not spoken word order; the TTS reads what it is
    # given, so the sentence has to already be the sentence a person would say.
    "2. Preštylizuj repliku tak, aby sa dobre hovorila nahlas: hovorový slovosled, "
    "krátke vety, register a slovník postavy z jej opisu nižšie.\n"
    # 3: canon turns these two marks into the model's own pause tokens, so timing
    # is written into the text instead of being chosen per line by a machine.
    "3. Kde má hovoriaci krátko zastať, napíš ' — '. Kde je pauza dlhšia, napíš '…'. "
    "Spolu najviac tri takéto značky.\n"
    # 4: the DM said what they meant; a "better" line the table never asked for is
    # the director mistake all over again.
    "4. Význam nemeň a nič nevymýšľaj: žiadne nové mená, čísla, fakty ani vety.\n"
    # 5: canon refuses anything longer, so a longer answer is a wasted round trip.
    f"5. Replika musí mať menej ako {canon.MAX_CHARS} znakov.\n"
    # 6: a delivery is the DM's choice, armed in the Lab; the Writer never picks one.
    "6. Nikdy nepíš značky v tvare <|...|>, javiskové poznámky, odrážky, úvodzovky "
    "okolo celej repliky ani vysvetlenia.\n"
    "Vráť LEN výslednú repliku ako jeden riadok.\n"
    # Two worked examples, because the other failure mode is a model that hands
    # the line back untouched when it cannot see what was wrong with it.
    "Príklady:\n"
    "vstup: cau kamos ako sa mas dnes rano\n"
    "výstup: Čau, kamoš. Ako sa máš dnes ráno?\n"
    "vstup: ten mec nechaj tam je prekliaty verim mi\n"
    "výstup: Ten meč tam nechaj — je prekliaty. Ver mi."
)

EN_FIX_SYSTEM = (
    "You are a proofreader and dialogue editor for a Dungeons & Dragons table. "
    "You are given one line that a character is about to say out loud.\n"
    # 0: see the Slovak prompt. The model must edit the line, never reply to it.
    "0. DO NOT ANSWER the line. You are not the character and you are not holding "
    "a conversation. You are given text and you return THAT SAME text, corrected. "
    "If the line is a question, you return the same question.\n"
    # 1: same first job as the Slovak prompt, minus the diacritics problem.
    "1. Fix spelling, grammar, agreement and word order.\n"
    # 2: the sentence has to be the sentence a person would actually say.
    "2. Rewrite it to be spoken well: spoken word order, short sentences, the "
    "register and vocabulary of the character described below.\n"
    # 3: canon maps these two marks to the model's own pause tokens.
    "3. Write ' — ' where the speaker takes a beat and '…' where the beat is "
    "longer. At most three such marks in total.\n"
    # 4: never invent; the DM's meaning is the line's meaning.
    "4. Do not change the meaning and do not invent anything: no new names, "
    "numbers, facts or sentences.\n"
    # 5: canon refuses anything longer.
    f"5. The line must be shorter than {canon.MAX_CHARS} characters.\n"
    # 6: deliveries belong to the DM, not to the Writer.
    "6. Never write <|...|> tokens, stage directions, bullet points, quotation "
    "marks around the whole line, or explanations.\n"
    "Return ONLY the resulting line, on one line.\n"
    "Examples:\n"
    "in: hey mate how are you doing this morning\n"
    "out: Hey, mate. How are you doing this morning?\n"
    "in: leave that sword its cursed trust me\n"
    "out: Leave that sword — it is cursed. Trust me."
)

_PERSONA_LEAD = {"sk": "Repliku hovorí táto postava:", "en": "The line is spoken by this character:"}


def residency() -> str:
    """``resident`` | ``loaded`` | ``absent`` for ``config.LLM_MODEL``.

    ``loaded`` means ollama holds the model but has spilled part of it to CPU:
    it would answer, several times too slowly for a DM waiting mid-scene, so the
    caller treats it exactly like ``absent``. Reported separately because the
    Lab and the status dot want to say *why* the brain is not ready.
    """
    try:
        payload = requests.get(config.LLM_URL + "/api/ps", timeout=PROBE_TIMEOUT_S).json()
    except (requests.RequestException, ValueError):
        return "absent"
    models = payload.get("models") or [] if isinstance(payload, dict) else []
    for m in models:
        if (m.get("name") or m.get("model")) != config.LLM_MODEL:
            continue
        size = m.get("size") or 0
        vram = m.get("size_vram") or 0
        return "resident" if size and vram / size >= RESIDENT_FRACTION else "loaded"
    return "absent"


def warming() -> bool:
    """True while a background load is in flight, so the UI can say "loading"
    instead of "absent" and the DM knows waiting is worth it."""
    return _warming.is_set()


def warm() -> str:
    """Load the model into VRAM and pin it there, then report the residency.

    WHY this exists: ``residency`` gates every Writer call, and ollama unloads a
    model after its idle timeout. Without a warm-up the brain would be absent
    forever -- nothing would ever ask for it, so nothing would ever load it.
    Blocking: the caller runs it off the request path (boot, or the wake button).
    """
    if _warming.is_set():
        return "loading"
    _warming.set()
    try:
        requests.post(config.LLM_URL + "/api/generate",
                      json={"model": config.LLM_MODEL, "prompt": "", "stream": False,
                            "keep_alive": -1, "options": {"num_ctx": NUM_CTX}},
                      timeout=WARM_TIMEOUT_S)
    except requests.RequestException as e:
        log.warning("brain warm-up failed: %s", e)
    finally:
        _warming.clear()
    state = residency()
    log.info("brain warm-up finished: %s", state)
    return state


def warm_in_background() -> None:
    """Fire and forget: boot must not wait a minute and a half for the brain."""
    if not _warming.is_set():
        threading.Thread(target=warm, name="brain-warm", daemon=True).start()


def fix(text: str, voice: Voice, lang: str = "sk") -> dict:
    """Correct the line and make it speakable, or hand it back untouched.

    Returns ``{"text", "original", "changed", "note"}``. ``original`` is always
    the text as typed, so the UI's undo restores it byte for byte. ``note`` is
    Slovak and non-empty whenever ``changed`` is False, because the pencil button
    must be able to say what happened instead of silently doing nothing.

    Raises ``BrainNotReady`` when the model is not resident; the API turns that
    into 503 and the button goes grey until the next ``status`` event.
    """
    original = text
    if residency() != "resident":
        raise BrainNotReady(config.LLM_MODEL)
    if not text.strip():
        return _unchanged(original, "prázdny text")
    result = _unchanged(original, "mozog neodpovedal")
    for attempt in range(ATTEMPTS):
        try:
            answer = _chat(text, voice, lang)
        except requests.Timeout:
            return _unchanged(original, "mozog neodpovedal do 8 sekúnd")
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            # A refused socket, a 500 or a body that is not the JSON ollama promises:
            # none of them may 500 the pencil, the DM's own line still speaks fine.
            return _unchanged(original, "mozog neodpovedal")
        result = _guard(original, answer, voice, lang)
        if result["changed"]:
            return result
        log.info("fix attempt %d rejected: %s", attempt + 1, result["note"])
    return result


def _chat(text: str, voice: Voice, lang: str) -> str:
    """One call against ``POST {LLM_URL}/api/chat``.

    ``fix`` calls this at most ``ATTEMPTS`` times, and only ever again after a
    guard has *rejected* an answer: re-rolling a usable line would buy a
    different sentence, not a better one. ``keep_alive: -1`` keeps the 27B
    pinned (a cold reload is ~90 s), ``think: false`` stops Qwen3 from spending
    the answer on a reasoning trace."""
    body = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": _system(voice, lang)},
                     {"role": "user", "content": text}],
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    r = requests.post(config.LLM_URL + "/api/chat", json=body, timeout=CHAT_TIMEOUT_S)
    r.raise_for_status()
    payload = r.json()
    return str((payload.get("message") or {}).get("content") or "") if isinstance(payload, dict) else ""


def _system(voice: Voice, lang: str) -> str:
    """The language's prompt plus the character's own persona, when the voice has
    one in that language. WHY only that language: a persona in the other language
    is what makes the model answer in a mix of both."""
    key = "en" if lang.lower().startswith("en") else "sk"
    base = EN_FIX_SYSTEM if key == "en" else SK_FIX_SYSTEM
    persona = str((voice.persona or {}).get(key) or "").strip()
    return f"{base}\n\n{_PERSONA_LEAD[key]} {persona}" if persona else base


def _guard(original: str, answer: str, voice: Voice, lang: str) -> dict:
    """Every promise the prompt made, re-checked here. Repairs come first (a
    stray token or a fourth beat is not worth throwing a good rewrite away),
    then the two checks that can only end in a refusal: a line whose length ran
    away is a line the model rewrote instead of fixed, and a line canon refuses
    could never be rendered anyway."""
    fixed = _clean(answer)
    if not fixed:
        return _unchanged(original, "mozog vrátil prázdnu odpoveď")
    ratio = len(fixed) / max(1, len(original.strip()))
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        return _unchanged(original, "oprava príliš zmenila dĺžku repliky")
    try:
        canon.canonicalize(fixed, lang=lang, banned=set(voice.banned_tokens))
    except canon.TooLong:
        return _unchanged(original, f"oprava je dlhšia ako {canon.MAX_CHARS} znakov")
    except (canon.BannedToken, ValueError):
        return _unchanged(original, "oprava neprešla kontrolou textu")
    if fixed == original:
        return _unchanged(original, "replika je v poriadku")
    return {"text": fixed, "original": original, "changed": True, "note": ""}


def _clean(answer: str) -> str:
    """Strip what the model adds around the line: a reasoning trace, control
    tokens (a delivery is the DM's choice, armed in the Lab, never the Writer's),
    quotes wrapped around the whole reply, and beats past the cap. Asterisks
    survive: canon reads ``Brá**cho`` as a drawl the DM asked for."""
    text = _THINK.sub("", answer)
    text = _TOKEN.sub(" ", text)
    text = " ".join(text.split()).strip(_QUOTES).strip()
    return _cap_beats(text)


def _cap_beats(text: str) -> str:
    """Keep the first ``MAX_BEATS`` beat marks and drop the rest: past three,
    the timing stops reading as timing and starts reading as a stall (canon
    drops the extra pause tokens for the same reason)."""
    seen = 0

    def keep(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        if seen > MAX_BEATS:
            return " "
        mark = match.group(0).strip()
        return f" {mark} " if mark in _SPACED_BEATS else f"{mark} "

    return " ".join(_BEAT.sub(keep, text).split())


def _unchanged(original: str, note: str) -> dict:
    """The safe answer: the DM's own text, and a reason they can read."""
    return {"text": original, "original": original, "changed": False, "note": note}


# -- the Voice Creator's two authoring calls (M4 contract, section 4) --------
#
# ``fix`` corrects one line the DM typed; these two invent text for a voice that
# does not exist yet. Same model, same residency rule, same "one attempt, then
# live with the answer" discipline, and the same principle that nothing the
# model says is trusted: every generated line is re-checked in code and dropped
# if it fails, because a soundboard is written once and then spoken at the table
# for months. Only the temperature differs (an author, not a corrector) and the
# timeout, which is measured against a DM sitting on the Creator page rather
# than mid-scene.

WRITE_TEMPERATURE = 0.8     # a corrector wants the same answer twice; an author wants variety
PERSONA_TIMEOUT_S = 60.0
PHRASES_TIMEOUT_S = 90.0    # one category of ten lines, on a box that is also rendering
MAX_PERSONA_CHARS = 700     # this text is prepended to every ``fix`` prompt; longer would eat num_ctx
MIN_CATEGORIES = 2          # a soundboard needs at least one ordinary tab plus the signature refusal
MAX_CATEGORIES = 8          # the Play page's tab strip, and 8 x 10 lines is already a long commit
MIN_LINES = 3               # under this the category is thin and the Creator page warns
MAX_LINES = 20
MAX_CATEGORY_CHARS = 40

_FENCE = re.compile(r"^```(?:json)?|```$", re.M)


class WriterFailed(RuntimeError):
    """The brain answered but nothing usable survived the guards.

    Distinct from ``BrainNotReady``: the model is resident, so retrying the same
    call is pointless and the DM has to be told rather than made to wait. ``fix``
    has no equivalent because it can always fall back to the DM's own text, and
    these two calls have nothing to fall back to.
    """


SK_PERSONA_SYSTEM = (
    "Si autor postáv pre slovenský stôl Dungeons & Dragons. Dostaneš meno postavy "
    "a jej krátky opis od rozprávača.\n"
    # The persona is prompt text: it is prepended to every later ``fix`` call, so
    # it has to describe how the character speaks, not what happened to them.
    "1. Napíš opis postavy pre hlasový model: register, tempo, typické slová, "
    "postoj k družine. Píš o tom, AKO hovorí, nie o jej príbehu. Najviac 5 viet.\n"
    "2. To isté napíš aj po anglicky, rovnako dlho.\n"
    # The categories become the soundboard's tabs, so they are what the DM
    # reaches for mid-scene: short, concrete, and named the way a DM thinks.
    "3. Navrhni {n} kategórií replík pre soundboard. Každá je krátky slovenský "
    "názov (2-4 slová) toho, čo postava hovorí v jednej situácii.\n"
    # The signature refusal is the giant red tile; it must exist and it must be
    # last, because that is where the board looks for it.
    "4. POSLEDNÁ kategória je vždy to, ako táto postava odmieta - jej vlastná "
    "hláška, nie všeobecné „Odmietnutie“.\n"
    'Vráť LEN JSON: {{"sk": "...", "en": "...", "categories": ["...", "..."]}}'
).format(n=MAX_CATEGORIES)

EN_PERSONA_SYSTEM = (
    "You are a character author for a Dungeons & Dragons table. You are given a "
    "character name and a short description from the DM.\n"
    "1. Write a character description for a voice model: register, pace, typical "
    "words, attitude to the party. Write about HOW they speak, not their backstory. "
    "At most 5 sentences.\n"
    "2. Write the same in Slovak, the same length.\n"
    "3. Propose {n} soundboard categories. Each is a short English name (2-4 "
    "words) for what the character says in one situation.\n"
    "4. The LAST category is always how this character refuses - their own line, "
    "not a generic \"Refusal\".\n"
    'Return ONLY JSON: {{"sk": "...", "en": "...", "categories": ["...", "..."]}}'
).format(n=MAX_CATEGORIES)

SK_PHRASES_SYSTEM = (
    "Si autor replík pre slovenský stôl Dungeons & Dragons. Píšeš repliky pre "
    "jednu postavu a jednu kategóriu soundboardu.\n"
    "1. Každá replika je jedna veta alebo dve, ktoré postava povie NAHLAS.\n"
    # canon refuses anything longer, so a longer line is a wasted round trip.
    "2. Najviac {n} znakov na repliku, hovorový slovosled, správna diakritika, "
    "register a slovník postavy.\n"
    "3. Žiadne mená hráčov, žiadne konkrétne miesta ani čísla z kampane - "
    "repliky musia sadnúť do každej scény.\n"
    # A delivery belongs to the DM and is armed in the Lab; a token written here
    # would be dropped by the guard anyway.
    "4. Nikdy nepíš značky <|...|>, javiskové poznámky, mená hovoriaceho, "
    "odrážky ani čísla riadkov.\n"
    "5. Repliky sa nesmú opakovať ani parafrázovať.\n"
    'Vráť LEN JSON: {{"lines": ["...", "..."]}}'
).format(n=canon.MAX_CHARS)

EN_PHRASES_SYSTEM = (
    "You are a dialogue author for a Dungeons & Dragons table. You write lines "
    "for one character and one soundboard category.\n"
    "1. Every line is one or two sentences the character says OUT LOUD.\n"
    "2. At most {n} characters per line, spoken word order, the register and "
    "vocabulary of the character.\n"
    "3. No player names, no specific places or numbers from the campaign - the "
    "lines must fit any scene.\n"
    "4. Never write <|...|> tokens, stage directions, speaker names, bullet "
    "points or line numbers.\n"
    "5. No line may repeat or paraphrase another.\n"
    'Return ONLY JSON: {{"lines": ["...", "..."]}}'
).format(n=canon.MAX_CHARS)

_CATEGORY_LEAD = {"sk": "Kategória", "en": "Category"}
_COUNT_LEAD = {"sk": "Napíš", "en": "Write"}


def persona(label: str, description: str, lang: str = "sk") -> dict:
    """A character sheet for a voice that does not exist yet.

    Returns ``{"sk", "en", "label", "description", "categories"}``: the two
    persona strings that go into ``voice.yaml`` (both languages, because a voice
    is spoken in both and a persona in the wrong language is what makes the model
    answer in a mix), the DM's own label and description echoed back so the
    Creator page can show what produced this, and the soundboard's category list
    whose **last entry is the signature refusal**.

    Raises ``BrainNotReady`` when the model is not resident and ``WriterFailed``
    when it answers with nothing usable; there is no fallback text, because an
    invented persona would quietly become the character.
    """
    if residency() != "resident":
        raise BrainNotReady(config.LLM_MODEL)
    key = _lang_key(lang)
    system = EN_PERSONA_SYSTEM if key == "en" else SK_PERSONA_SYSTEM
    answer = _object(_ask(system, f"{label}\n\n{description}".strip(), PERSONA_TIMEOUT_S))
    sk, en = _persona_text(answer.get("sk")), _persona_text(answer.get("en"))
    categories = _categories(answer.get("categories"))
    if not (sk or en) or len(categories) < MIN_CATEGORIES:
        raise WriterFailed("persona reply carried no usable text or too few categories")
    # A voice with only one persona would lose its character in the other
    # language, so whichever came back stands in for the missing one.
    return {"sk": sk or en, "en": en or sk, "label": label, "description": description,
            "categories": categories}


def phrases(persona: dict, categories: list[str], lang: str = "sk",
            per_category: int = 10) -> dict[str, list[str]]:
    """The soundboard: up to ``per_category`` usable lines for every category.

    One call per category, not one call for the whole board: ``num_ctx`` is 4096
    and a board is eight categories deep, and a single thin category can then be
    re-asked on its own instead of re-rolling lines the DM already liked. A
    category still under ``MIN_LINES`` after that one retry is left thin and
    logged; the Creator page shows the count and warns, because three real tiles
    beat ten padded ones.
    """
    if residency() != "resident":
        raise BrainNotReady(config.LLM_MODEL)
    key = _lang_key(lang)
    lead = str(persona.get(key) or persona.get("sk") or persona.get("en") or "").strip()
    board: dict[str, list[str]] = {}
    for category in categories:
        lines = _category_lines(lead, category, key, per_category, lang)
        if len(lines) < MIN_LINES:
            lines = _merge(lines, _category_lines(lead, category, key, per_category, lang), per_category)
        if len(lines) < MIN_LINES:
            log.warning("thin category %r: %d usable lines", category, len(lines))
        board[category] = lines
    return board


def _category_lines(lead: str, category: str, key: str, per_category: int, lang: str) -> list[str]:
    """One attempt at one category, with every line already through the guards."""
    system = EN_PHRASES_SYSTEM if key == "en" else SK_PHRASES_SYSTEM
    want = max(1, min(int(per_category), MAX_LINES))
    ask = (f"{_PERSONA_LEAD[key]} {lead}\n\n{_CATEGORY_LEAD[key]}: {category}\n"
           f"{_COUNT_LEAD[key]} {want}.")
    try:
        answer = _ask(system, ask, PHRASES_TIMEOUT_S)
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        # A refused socket or a body that is not the JSON ollama promises costs
        # this category its lines, not the whole board: the rest are still worth
        # writing, and the DM sees the empty tab and can re-ask it alone.
        log.warning("category %r: the brain did not answer", category)
        return []
    return _merge([], [usable_line(item, lang) for item in _lines(answer)], want)


def _ask(system: str, user: str, timeout: float) -> str:
    """One attempt against ``POST {LLM_URL}/api/chat``, on the same terms as
    ``_chat``: no retry, ``think: false`` so the answer is the answer and not a
    reasoning trace, ``keep_alive: -1`` so the 27B stays pinned between the
    Creator's steps (a cold reload is ~90 s and a commit makes ten of these
    calls in a row), ``num_ctx`` 4096 because a persona plus one category fits."""
    body = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": {"temperature": WRITE_TEMPERATURE, "num_ctx": NUM_CTX},
    }
    r = requests.post(config.LLM_URL + "/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    return str((payload.get("message") or {}).get("content") or "") if isinstance(payload, dict) else ""


def _lang_key(lang: str) -> str:
    return "en" if str(lang).lower().startswith("en") else "sk"


def _json(answer: str) -> object | None:
    """The JSON hiding in the model's reply, or ``None``.

    Defensive on purpose: a mid-size model asked for JSON still wraps it in a
    fence, prefixes "Here you go:" or appends a paragraph of commentary. The
    outermost brace (or bracket) pair is taken and parsed; anything else is a
    miss, which the callers read as an empty answer rather than an exception.
    """
    text = _FENCE.sub("", _THINK.sub("", answer)).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    return None


def _object(answer: str) -> dict:
    parsed = _json(answer)
    return parsed if isinstance(parsed, dict) else {}


def _lines(answer: str) -> list[object]:
    """The model's line list, whether or not it wrapped it in an object."""
    parsed = _json(answer)
    if isinstance(parsed, dict):
        for key in ("lines", "phrases", "repliky"):
            if isinstance(parsed.get(key), list):
                return list(parsed[key])
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _persona_text(value: object) -> str:
    """Prompt text, not a spoken line: reasoning traces and control tokens go,
    and the length is capped so a persona cannot crowd the DM's own line out of
    ``fix``'s context window."""
    if not isinstance(value, str):
        return ""
    text = " ".join(_TOKEN.sub(" ", _THINK.sub("", value)).split()).strip(_QUOTES).strip()
    return text[:MAX_PERSONA_CHARS].strip()


def _categories(value: object) -> list[str]:
    """Tab names: short, unique, kept in the model's order, because the last one
    is the signature refusal and only the order says which one that is."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        name = _persona_text(item)[:MAX_CATEGORY_CHARS].strip(" .:-")
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            out.append(name)
    return out[:MAX_CATEGORIES]


def usable_line(value: object, lang: str) -> str | None:
    """One generated line, or ``None`` when it may not become a tile.

    Dropped rather than repaired, unlike ``fix``: no DM is waiting on this
    particular sentence, so a line carrying a control token (a delivery is the
    DM's choice, armed in the Lab), running past ``canon.MAX_CHARS`` or refused
    by canon is simply not worth keeping while nine others are fine.
    """
    if not isinstance(value, str):
        return None
    line = " ".join(_THINK.sub("", value).split()).strip(_QUOTES).strip()
    if not line or _TOKEN.search(line) or len(line) > canon.MAX_CHARS:
        return None
    try:
        canon.canonicalize(line, lang=lang)
    except (canon.TooLong, canon.BannedToken, ValueError):
        return None
    return line


def _merge(kept: list[str], extra: list[str | None], limit: int) -> list[str]:
    """Append what survived the guards, without duplicates, up to ``limit``."""
    out = list(kept)
    seen = {line.casefold() for line in out}
    for line in extra:
        if line and line.casefold() not in seen and len(out) < limit:
            seen.add(line.casefold())
            out.append(line)
    return out
