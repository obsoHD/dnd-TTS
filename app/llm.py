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

import re

import requests

from app import canon, config
from app.voices import Voice

RESIDENT_FRACTION = 0.9     # below this ollama has spilled the model to CPU: too slow for the table
PROBE_TIMEOUT_S = 3.0       # WHY shorter than the chat call: the pencil must go grey fast, not hang
CHAT_TIMEOUT_S = 8.0
TEMPERATURE = 0.3           # a corrector, not an author: the same line twice should come back the same
NUM_CTX = 4096              # one line plus a persona; a 40k context would waste ~8 GB of VRAM
MAX_BEATS = 3               # canon caps pause tokens at 3 too, so a fourth beat would be dropped anyway
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
    "Dostaneš jednu repliku, ktorú niekto o chvíľu povie nahlas.\n"
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
    "Vráť LEN výslednú repliku ako jeden riadok."
)

EN_FIX_SYSTEM = (
    "You are a proofreader and dialogue editor for a Dungeons & Dragons table. "
    "You are given one line that someone is about to say out loud.\n"
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
    "Return ONLY the resulting line, on one line."
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
    try:
        answer = _chat(text, voice, lang)
    except requests.Timeout:
        return _unchanged(original, "mozog neodpovedal do 8 sekúnd")
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        # A refused socket, a 500 or a body that is not the JSON ollama promises:
        # none of them may 500 the pencil, the DM's own line still speaks fine.
        return _unchanged(original, "mozog neodpovedal")
    return _guard(original, answer, voice, lang)


def _chat(text: str, voice: Voice, lang: str) -> str:
    """One attempt against ``POST {LLM_URL}/api/chat``. No retry: a second roll
    of the dice costs the table another 8 s and buys a different sentence, not a
    better one. ``keep_alive: -1`` keeps the 27B pinned (a cold reload is ~90 s),
    ``think: false`` stops Qwen3 from spending the answer on a reasoning trace."""
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
