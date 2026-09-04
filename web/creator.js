/**
 * Voice Creator: five steps from a raw clip to a table-ready voice.
 *
 * The server owns the draft. Every step posts to `/api/creator/...`, takes the
 * Draft that comes back and mirrors it into the form fields, so a reload (or a
 * second tablet) resumes exactly where the DM left off instead of trusting
 * anything this page remembers. The only thing kept in the browser is the
 * draft id, and even that is only a hint: a 404 on it starts a fresh one.
 *
 * The five steps stay stacked in one column with the reached ones collapsed to
 * a summary line, because "no wizard magic" means the DM can always go back and
 * see what a finished step decided, not only what the next one wants.
 *
 * The socket/api helpers are copies of app.js's on purpose: app.js exports
 * nothing and belongs to another builder, so this page carries the small
 * amount it needs rather than reaching into theirs.
 */
import { h, render, Component } from './vendor/preact.min.js';
import htm from './vendor/htm.js';

const html = htm.bind(h);
const audio = document.getElementById('audio');

const DRAFT_KEY = 'bag.creator.draft';
const MAX_UPLOAD = 25 * 1024 * 1024;          // the API's cap; checked here so a 25 MB upload is not sent to be refused
const MAX_CLIP_S = 30;                        // voices.lock_reference truncates past this; the DM picks the window instead
const THIN = 3;                               // a category under three usable lines is reported thin by the writer
const ID_RE = /^[a-z0-9_-]{2,24}$/;
const RESERVED = new Set(['bag', 'male', 'female', 'shopkeep']);
const LANGS = ['sk', 'en'];
const BRAIN_DOWN = 'mozog nie je pripravený';
const STEPS = [
  { n: 1, title: 'Klip' },
  { n: 2, title: 'Prepis' },
  { n: 3, title: 'Postava' },
  { n: 4, title: 'Frázy' },
  { n: 5, title: 'Vytvorenie' },
];
/** Draft.status -> the furthest step it unlocks. Content can push it further; nothing pulls it back. */
const STATUS_STEP = { new: 1, clipped: 2, transcribed: 3, described: 4, locked: 5, calibrated: 5, banked: 5, done: 5, failed: 5 };

/* ---------------- helpers (copied from app.js: it exports nothing) ---------------- */

/** Stable per-browser id so the hub can tell tablets apart across reloads; own key so the Play page stays a separate client. */
function clientId() {
  const key = 'bag.client.creator';
  let id = null;
  try { id = localStorage.getItem(key); } catch { /* private mode: a per-load id is fine */ }
  if (!id) {
    id = Math.random().toString(36).slice(2, 10);
    try { localStorage.setItem(key, id); } catch { /* ignore */ }
  }
  return id;
}

const wsUrl = (id) => `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?client=${id}&role=play`;

/** The API answers 400 with the reason (bad window, duplicate id, failed conversion): keep it, it is the whole message. */
async function httpError(res, method, path) {
  let detail = null;
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') detail = body.detail;
  } catch { /* not JSON: the status line is all we have */ }
  const err = new Error(detail || `${method} ${path} -> ${res.status}`);
  err.status = res.status;
  return err;
}

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw await httpError(res, method, path);
  return res.status === 204 ? null : res.json();
}

/** Multipart: the browser sets the boundary, so no content-type header here. */
async function upload(path, form) {
  const res = await fetch(path, { method: 'POST', body: form });
  if (!res.ok) throw await httpError(res, 'POST', path);
  return res.json();
}

/** Accept {event,data} as well as flat {type,...} frames: the hub's wire shape is not pinned by the contract. */
function parseFrame(raw) {
  let m;
  try { m = JSON.parse(raw); } catch { return null; }
  const event = m.event ?? m.type;
  if (!event) return null;
  if (m.data !== undefined) return { event, data: m.data };
  return { event, data: Object.fromEntries(Object.entries(m).filter(([k]) => k !== 'event' && k !== 'type')) };
}

/** WebSocket that reconnects forever with capped exponential backoff and jitter. */
function openSocket(url, on) {
  let ws;
  let attempt = 0;
  const connect = () => {
    ws = new WebSocket(url);
    ws.onopen = () => { attempt = 0; on.open(); };
    ws.onmessage = (m) => { const f = parseFrame(m.data); if (f) on.event(f.event, f.data); };
    ws.onerror = () => ws.close();
    ws.onclose = () => {
      on.close();
      const wait = Math.min(8000, 500 * 2 ** attempt++) * (0.75 + Math.random() / 2);
      setTimeout(connect, wait);
    };
  };
  connect();                       // nothing is ever sent from here: the creator only listens
}

/* ---------------- draft shape ---------------- */

const num = (v) => (Number.isFinite(+v) ? +v : 0);
const catsOf = (d) => (Array.isArray(d?.categories) && d.categories.length ? d.categories : Object.keys(d?.phrases || {}));
const linesOf = (phrases, cat) => (Array.isArray(phrases?.[cat]) ? phrases[cat] : []);
const lineCount = (d) => Object.values(d?.phrases || {}).reduce((n, v) => n + (Array.isArray(v) ? v.length : 0), 0);
const band = (d) => (Array.isArray(d?.f0_band) && d.f0_band.length === 2 && d.f0_band[0] > 0 ? d.f0_band : null);

/** How far the DM may click. Status is the server's word; transcript and phrases can only push it further. */
function reach(d) {
  if (!d) return 1;
  let step = STATUS_STEP[d.status] ?? 1;
  if ((d.transcript || '').trim()) step = Math.max(step, 3);
  if (lineCount(d)) step = Math.max(step, 5);
  return Math.min(STEPS.length, Math.max(1, step));
}

/** The persona is per language plus an echo of label/description; only the language's text is edited here. */
function personaText(persona, lang) {
  const v = persona?.[lang] ?? persona?.sk ?? persona?.en ?? '';
  return typeof v === 'string' ? v : JSON.stringify(v, null, 2);
}

function idError(id) {
  if (!id) return 'zadaj id hlasu';
  if (!ID_RE.test(id)) return 'id hlasu: 2-24 znakov, len a-z, 0-9, _ a -';
  if (RESERVED.has(id)) return `id "${id}" je vyhradené pre existujúci hlas`;
  return null;
}

/** A default id from the label: diacritics folded, everything else to a dash, because the id is a filesystem path. */
function slug(label) {
  const flat = [...(label || '').normalize('NFD')]
    .filter((c) => c.codePointAt(0) < 0x300 || c.codePointAt(0) > 0x36f).join('');
  return flat.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 24);
}

function fileError(f) {
  if (!f) return 'vyber zvukový súbor';
  if (f.size > MAX_UPLOAD) return `súbor má ${(f.size / 1048576).toFixed(1)} MB, limit je 25 MB`;
  if (f.type && !f.type.startsWith('audio/')) return `typ ${f.type} nie je zvuk`;
  return null;
}

function windowError(start, end) {
  if (!(end > start)) return 'koniec musí byť za začiatkom';
  if (end - start > MAX_CLIP_S) return `okno má ${(end - start).toFixed(1)} s, maximum je ${MAX_CLIP_S} s`;
  return null;
}

const storeDraft = (id) => { try { localStorage.setItem(DRAFT_KEY, id); } catch { /* ignore */ } };
const clearDraft = () => { try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ } };
const storedDraft = () => { try { return localStorage.getItem(DRAFT_KEY); } catch { return null; } };
/** A shared link wins over this browser's memory: two tablets can then look at the same draft. */
const linkedDraft = () => new URLSearchParams(location.search).get('draft');

const pct = (v) => Math.max(0, Math.min(100, Math.round(num(v))));
/** An object with something in it, or null: an empty dict from the server means "not answered". */
const filled = (o) => (o && typeof o === 'object' && Object.keys(o).length ? o : null);

/* ---------------- pure components ---------------- */

const Spin = () => html`<span class="cr-spin"></span>`;

/** One step. Collapsed it is a summary line, so a finished step still says what it decided. */
const Step = ({ n, title, open, reached, summary, onOpen, children }) => html`
  <section class="cr-step ${open ? 'open' : ''} ${reached ? '' : 'locked'}">
    <button class="cr-head" disabled=${!reached} aria-expanded=${open ? 'true' : 'false'} onClick=${onOpen}>
      <span class="cr-num">${n}</span>
      <span class="cr-title">${title}</span>
      <span class="cr-sum">${reached ? summary : 'zatiaľ nedostupné'}</span>
    </button>
    ${open && html`<div class="cr-body">${children}</div>`}
  </section>`;

/** The hint line is always rendered, empty or not: fields of equal height are what lets a
 *  row bottom-align its inputs without the labels above them drifting apart. */
const Field = ({ label, hint, children }) => html`<label class="cr-field">
  <span class="cr-lbl">${label}</span>${children}<small class="cr-hint">${hint || ''}</small>
</label>`;

const Next = ({ ok, why, onClick }) => html`<div class="cr-foot">
  ${!ok && why && html`<span class="cr-hint">${why}</span>`}
  <button class="speak" disabled=${!ok} onClick=${onClick}>Ďalej</button>
</div>`;

const LangPills = ({ value, onPick }) => html`<span class="pills">${LANGS.map((l) => html`
  <button key=${l} class="pill ${l === value ? 'on' : ''}" onClick=${() => onPick(l)}>${l.toUpperCase()}</button>`)}</span>`;

/** Gate numbers as the Lab prints them: the DM should see the thresholds the new voice will be judged by. */
function GateCard({ result, onNew }) {
  const g = result.gate || {};
  const sim = (v) => (typeof v === 'number' ? v.toFixed(3) : v);      // similarities are read to three places
  const rows = [
    ['baseline', sim(g.baseline)], ['strict', sim(g.strict)], ['loose', sim(g.loose)], ['p10', sim(g.p10)],
    ['CER max', sim(g.cer_max)], ['gain dB', typeof result.gain_db === 'number' ? result.gain_db.toFixed(1) : result.gain_db],
  ].filter(([, v]) => v !== null && v !== undefined);
  return html`<div class="cr-done">
    <h3>Hlas <em>${result.voice_id}</em> je pripravený</h3>
    <dl class="cr-gate">${rows.map(([k, v]) => html`<div key=${k}><dt>${k}</dt><dd>${v}</dd></div>`)}</dl>
    <p class="cr-hint">
      ${result.lines != null ? `${result.lines} viet v banke` : 'banka naplnená'}
      ${result.queued != null ? ` / ${result.queued} v rendrovacej fronte` : ''}
      ${g.mode ? ` / režim brány: ${g.mode}` : ''}
    </p>
    ${!result.gate && html`<p class="cr-hint">Čísla brány sa zobrazia po najbližšom štarte služby.</p>`}
    <div class="cr-row">
      <a class="speak" href=${`/?voice=${encodeURIComponent(result.voice_id || '')}`}>Späť na Play</a>
      <button class="cr-btn ghost" onClick=${onNew}>Nový hlas</button>
    </div>
  </div>`;
}

/* ---------------- App ---------------- */

class Creator extends Component {
  state = {
    draft: null, drafts: [], step: 1, wsUp: false, busy: null, error: null,
    file: null, dragging: false, upLabel: '', upLang: 'sk',      // step 1, before a draft exists
    start: 0, end: 0, playing: false, clipRev: 0,                // step 1, after it
    transcript: '',                                              // step 2
    label: '', voiceId: '', lang: 'sk', description: '', persona: '', categories: [], newCat: '',   // step 3
    perCategory: 10, phrases: {}, newLine: {},                   // step 4
    jobId: null, progress: null, result: null, commitError: null, // step 5
  };
  id = clientId();

  componentDidMount() {
    audio.addEventListener('ended', () => this.setState({ playing: false }));
    audio.addEventListener('pause', () => this.setState({ playing: false }));
    openSocket(wsUrl(this.id), {
      open: () => this.setState({ wsUp: true }),
      close: () => this.setState({ wsUp: false }),
      event: (ev, d) => this.onEvent(ev, d),
    });
    this.boot();
  }

  /** The draft on the server is the truth; the stored id is only a hint, and a stale one just starts over. */
  async boot() {
    const drafts = await api('GET', '/api/creator/drafts').catch(() => []);
    this.setState({ drafts: Array.isArray(drafts) ? drafts : [] });
    const id = linkedDraft() || storedDraft();
    if (!id) return;
    const d = await api('GET', `/api/creator/${encodeURIComponent(id)}`).catch(() => null);
    if (!d) return clearDraft();
    this.adopt(d, true);
    // a reload after the commit lands here: the draft is done, so show the card instead of the button again
    if (d.status === 'done') this.finish(null, d);
  }

  /** Mirror a Draft into the form fields. `jump` opens the furthest reached step (boot, resume). */
  adopt(d, jump = false) {
    storeDraft(d.id);
    this.setState((s) => ({
      draft: d,
      step: jump ? reach(d) : Math.min(s.step, reach(d)),
      start: num(d.start_s), end: num(d.end_s),
      transcript: d.transcript || '',
      label: d.label || '', voiceId: d.voice_id || slug(d.label), lang: d.lang || 'sk',
      description: d.description || '', persona: personaText(d.persona, d.lang || 'sk'),
      categories: catsOf(d), phrases: { ...(d.phrases || {}) }, newLine: {},
    }));
  }

  fail(e) {
    this.setState({ error: e.status === 503 ? BRAIN_DOWN : e.message });
  }

  /** One call at a time: every long step (whisper, the Writer, the commit) blocks the others. */
  async run(key, fn) {
    if (this.state.busy) return null;
    this.setState({ busy: key, error: null });
    try {
      return await fn();
    } catch (e) {
      this.fail(e);
      return null;
    } finally {
      this.setState({ busy: null });
    }
  }

  go(step) {
    this.setState({ step, error: null }, () => {
      document.querySelector('.cr-step.open')?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    });
  }

  /* ---- events ---- */

  onEvent(ev, d) {
    const draftId = this.state.draft?.id;
    if (ev === 'creator.progress' && (!d.draft_id || d.draft_id === draftId)) {
      this.setState({ progress: { step: d.step || '', pct: pct(d.pct) } });
      if (d.result || d.gate) return this.finish(d.result || d);
      if (pct(d.pct) >= 100) return this.finish(null);
      return undefined;
    }
    const mine = this.state.jobId && (d.job_id === this.state.jobId || d.id === this.state.jobId);
    if (!mine && d.draft_id !== draftId) return undefined;
    if (ev === 'creator.done' || (ev === 'job.done' && mine)) return this.finish(d.result || d);
    if (ev === 'creator.failed' || (ev === 'job.failed' && mine)) {
      return this.setState({ commitError: d.error || 'vytvorenie hlasu zlyhalo', progress: null });
    }
    return undefined;
  }

  /** The commit result rides the socket; the voice record is the fallback for the gate numbers.
   *  `known` is the draft the caller already holds: boot calls this before its own setState has
   *  flushed, and reading a stale `this.state.draft` here would drop the draft on the floor. */
  async finish(result, known = null) {
    if (this.state.result) return;                     // the first completion wins; later echoes are noise
    this.setState({ progress: { step: 'hotovo', pct: 100 }, commitError: null });
    const base = known || this.state.draft;
    const fresh = base ? await api('GET', `/api/creator/${encodeURIComponent(base.id)}`).catch(() => null) : null;
    const voiceId = result?.voice_id || fresh?.voice_id || base?.voice_id || this.state.voiceId;
    const voice = filled(result?.gate) || !voiceId ? null
      : await api('GET', `/api/voices/${encodeURIComponent(voiceId)}`).catch(() => null);
    this.setState((s) => ({
      draft: fresh || base || s.draft,
      result: {
        voice_id: voiceId,
        // an uncalibrated voice answers with an empty gate: that is "no numbers yet", not numbers
        gate: filled(result?.gate) ?? filled(voice?.gate) ?? null,
        gain_db: result?.gain_db ?? voice?.master?.gain_db ?? null,
        lines: result?.lines ?? null,
        queued: result?.queued ?? null,
      },
    }));
  }

  /* ---- step 1: the clip ---- */

  pickFile(file) {
    const err = fileError(file);
    this.setState({
      file: err ? null : file, dragging: false, error: err,
      upLabel: !err && !this.state.upLabel ? file.name.replace(/\.[^.]+$/, '') : this.state.upLabel,
    });
  }

  create() {
    const { file, upLabel, upLang } = this.state;
    const err = fileError(file) || (upLabel.trim() ? null : 'zadaj meno postavy');
    if (err) return this.setState({ error: err });
    return this.run('create', async () => {
      const form = new FormData();
      form.append('file', file, file.name);
      form.append('label', upLabel.trim());
      form.append('lang', upLang);
      this.adopt(await upload('/api/creator/drafts', form));
    });
  }

  trim() {
    const { draft, start, end } = this.state;
    const err = windowError(start, end);
    if (err) return this.setState({ error: err });
    return this.run('trim', async () => {
      const d = await api('POST', `/api/creator/${encodeURIComponent(draft.id)}/clip`, { start_s: start, end_s: end });
      this.adopt(d);
      this.setState((s) => ({ clipRev: s.clipRev + 1 }));   // the URL is stable, so the cache needs busting
    });
  }

  /** Audition the trimmed clip: this is the only place the DM hears what the clone will imitate. */
  toggleClip() {
    const d = this.state.draft;
    if (!d) return;
    if (!audio.paused) return audio.pause();
    audio.src = `/api/creator/${encodeURIComponent(d.id)}/clip.wav?v=${this.state.clipRev}`;
    audio.play().then(() => this.setState({ playing: true }))
      .catch(() => this.setState({ error: 'klip sa nedá prehrať' }));
  }

  /* ---- steps 2-4: transcript, character, phrases ---- */

  transcribe() {
    const d = this.state.draft;
    return this.run('transcribe', async () => this.adopt(await api('POST', `/api/creator/${encodeURIComponent(d.id)}/transcribe`)));
  }

  /** Every save goes through PATCH and re-reads the Draft, so what is on screen is what the server stored. */
  patch(key, body) {
    const d = this.state.draft;
    if (!d) return null;
    return this.run(key, async () => this.adopt(await api('PATCH', `/api/creator/${encodeURIComponent(d.id)}`, body)));
  }

  saveIdentity() {
    const { label, voiceId, lang } = this.state;
    const err = idError(voiceId) || (label.trim() ? null : 'zadaj meno postavy');
    if (err) return this.setState({ error: err });
    return this.patch('identity', { label: label.trim(), voice_id: voiceId, lang });
  }

  describe() {
    const { description } = this.state;
    if (!description.trim()) return this.setState({ error: 'napíš pár viet o postave' });
    return this.patch('describe', { description: description.trim() });
  }

  savePersona() {
    const { draft, persona, lang } = this.state;
    return this.patch('persona', { persona: { ...(draft?.persona || {}), [lang]: persona } });
  }

  saveCategories(categories) {
    this.setState({ categories });
    return this.patch('categories', { categories });
  }

  writePhrases() {
    const d = this.state.draft;
    return this.run('phrases', async () => {
      const body = { per_category: Math.max(1, Math.round(num(this.state.perCategory))) };
      this.adopt(await api('POST', `/api/creator/${encodeURIComponent(d.id)}/phrases`, body));
    });
  }

  /** Local first so typing stays smooth; the PATCH lands on blur (onChange) and the answer becomes the truth. */
  editLine(cat, i, text) {
    this.setState((s) => {
      const lines = [...linesOf(s.phrases, cat)];
      lines[i] = text;
      return { phrases: { ...s.phrases, [cat]: lines } };
    });
  }

  dropLine(cat, i) {
    const lines = linesOf(this.state.phrases, cat).filter((_, j) => j !== i);
    const phrases = { ...this.state.phrases, [cat]: lines };
    this.setState({ phrases });
    this.patch('phrases-save', { phrases });
  }

  addLine(cat) {
    const text = (this.state.newLine[cat] || '').trim();
    if (!text) return;
    const phrases = { ...this.state.phrases, [cat]: [...linesOf(this.state.phrases, cat), text] };
    this.setState((s) => ({ phrases, newLine: { ...s.newLine, [cat]: '' } }));
    this.patch('phrases-save', { phrases });
  }

  savePhrases() { return this.patch('phrases-save', { phrases: this.state.phrases }); }

  /* ---- step 5: commit ---- */

  commit() {
    const d = this.state.draft;
    return this.run('commit', async () => {
      this.setState({ commitError: null, progress: { step: 'štart', pct: 0 } });
      const r = await api('POST', `/api/creator/${encodeURIComponent(d.id)}/commit`);
      this.setState({ jobId: r?.job_id ?? null });
    });
  }

  /** Back to an empty step 1. The committed draft stays on the server: it is the record of how
   *  this voice was made, and only the DM's explicit Zahodiť removes one. */
  reset() {
    clearDraft();
    this.setState({
      draft: null, step: 1, file: null, upLabel: '', transcript: '', description: '', persona: '',
      categories: [], phrases: {}, newLine: {}, result: null, jobId: null, progress: null,
      commitError: null, error: null,
    }, () => this.boot());
  }

  discard() {
    const d = this.state.draft;
    if (!d || !window.confirm(`Zahodiť rozpracovaný hlas ${d.label || d.id}?`)) return null;
    return this.run('discard', async () => {
      await api('DELETE', `/api/creator/${encodeURIComponent(d.id)}`);
      this.reset();
    });
  }

  /* ---- render ---- */

  renderClip(d) {
    const { start, end, busy, playing } = this.state;
    const b = band(d);
    const err = windowError(start, end);
    return html`
      <p class="cr-hint">Vyber okno bez hudby a bez reakcií stola: čokoľvek cudzie v klipe sa naklonuje spolu s hlasom.</p>
      <div class="cr-row">
        <${Field} label="začiatok (s)">
          <input class="cr-input" type="number" min="0" step="0.1" value=${start}
            onInput=${(e) => this.setState({ start: num(e.target.value) })} />
        <//>
        <${Field} label="koniec (s)">
          <input class="cr-input" type="number" min="0" step="0.1" value=${end}
            onInput=${(e) => this.setState({ end: num(e.target.value) })} />
        <//>
        <${Field} label="dĺžka" hint=${`maximum ${MAX_CLIP_S} s`}>
          <output class="cr-out ${err ? 'bad' : ''}">${(end - start).toFixed(1)} s</output>
        <//>
      </div>
      <div class="cr-row">
        <button class="cr-btn" onClick=${() => this.toggleClip()}>${playing ? 'Pauza' : 'Prehrať klip'}</button>
        <button class="cr-btn go" disabled=${!!err || busy === 'trim'} onClick=${() => this.trim()}>
          ${busy === 'trim' ? html`<${Spin} />` : 'Orezať'}
        </button>
        <span class="cr-band">${b ? `výška hlasu ${b[0]}-${b[1]} Hz` : 'výška hlasu sa zmeria pri orezaní'}</span>
      </div>
      <${Next} ok=${reach(d) >= 2} why="najprv orež klip" onClick=${() => this.go(2)} />`;
  }

  renderUpload() {
    const { file, dragging, upLabel, upLang, busy, drafts } = this.state;
    return html`
      <div class="cr-drop ${dragging ? 'over' : ''}"
        onDragOver=${(e) => { e.preventDefault(); this.setState({ dragging: true }); }}
        onDragLeave=${() => this.setState({ dragging: false })}
        onDrop=${(e) => { e.preventDefault(); this.pickFile(e.dataTransfer?.files?.[0]); }}>
        <input class="cr-file" type="file" accept="audio/*" onChange=${(e) => this.pickFile(e.target.files[0])} />
        <span class="cr-hint">${file ? `${file.name} / ${(file.size / 1048576).toFixed(1)} MB` : 'pretiahni sem nahrávku alebo ju vyber (max 25 MB)'}</span>
      </div>
      <div class="cr-row">
        <${Field} label="meno postavy">
          <input class="cr-input" type="text" value=${upLabel} placeholder="napr. Kováč Radomír"
            onInput=${(e) => this.setState({ upLabel: e.target.value })} />
        <//>
        <${Field} label="jazyk">
          <${LangPills} value=${upLang} onPick=${(l) => this.setState({ upLang: l })} />
        <//>
      </div>
      <div class="cr-row">
        <button class="cr-btn go" disabled=${!file || busy === 'create'} onClick=${() => this.create()}>
          ${busy === 'create' ? html`<${Spin} />` : 'Nahrať'}
        </button>
      </div>
      ${!!drafts.length && html`<div class="cr-drafts">
        <span class="cr-hint">rozpracované:</span>
        ${drafts.map((x) => html`<a key=${x.id} class="tab" href=${`?draft=${encodeURIComponent(x.id)}`}>${x.label || x.id} (${x.status})</a>`)}
      </div>`}`;
  }

  renderTranscript(d) {
    const { transcript, busy } = this.state;
    return html`
      <p class="cr-hint">Prepis musí sedieť s klipom slovo za slovom. Zlý prepis je najčastejšia príčina rozpadnutého hlasu.</p>
      <textarea class="cr-area" rows="5" value=${transcript} spellcheck="false"
        onInput=${(e) => this.setState({ transcript: e.target.value })}></textarea>
      <div class="cr-row">
        <button class="cr-btn" disabled=${busy === 'transcribe'} onClick=${() => this.transcribe()}>
          ${busy === 'transcribe' ? html`<${Spin} />` : d.transcript ? 'Prepísať znova' : 'Prepísať (whisper)'}
        </button>
        <button class="cr-btn go" disabled=${!transcript.trim() || busy === 'transcript'} onClick=${() => this.patch('transcript', { transcript })}>
          ${busy === 'transcript' ? html`<${Spin} />` : 'Uložiť prepis'}
        </button>
      </div>
      <${Next} ok=${!!(d.transcript || '').trim()} why="ulož opravený prepis" onClick=${() => this.go(3)} />`;
  }

  renderCharacter(d) {
    const { label, voiceId, lang, description, persona, categories, newCat, busy } = this.state;
    const err = idError(voiceId);
    return html`
      <div class="cr-row">
        <${Field} label="meno postavy">
          <input class="cr-input" type="text" value=${label} onInput=${(e) => this.setState({ label: e.target.value })} />
        <//>
        <${Field} label="id hlasu" hint=${err || 'používa sa v ceste a v cache, už sa nemení'}>
          <input class="cr-input ${err ? 'bad' : ''}" type="text" value=${voiceId} autocomplete="off"
            onInput=${(e) => this.setState({ voiceId: e.target.value.trim().toLowerCase() })} />
        <//>
        <${Field} label="jazyk">
          <${LangPills} value=${lang} onPick=${(l) => this.setState({ lang: l })} />
        <//>
      </div>
      <div class="cr-row">
        <button class="cr-btn" disabled=${busy === 'identity'} onClick=${() => this.saveIdentity()}>
          ${busy === 'identity' ? html`<${Spin} />` : 'Uložiť'}
        </button>
      </div>
      <${Field} label="opis postavy" hint="pár viet: kto to je, ako hovorí, čo robí pri stole">
        <textarea class="cr-area" rows="4" value=${description}
          onInput=${(e) => this.setState({ description: e.target.value })}></textarea>
      <//>
      <div class="cr-row">
        <button class="cr-btn go" disabled=${!description.trim() || busy === 'describe'} onClick=${() => this.describe()}>
          ${busy === 'describe' ? html`<${Spin} />` : 'Generovať'}
        </button>
        <span class="cr-hint">Writer vyplní personu a zoznam kategórií; oboje sa dá prepísať.</span>
      </div>
      ${(persona || categories.length) ? html`
        <${Field} label=${`persona (${lang})`}>
          <textarea class="cr-area" rows="6" value=${persona}
            onInput=${(e) => this.setState({ persona: e.target.value })}
            onChange=${() => this.savePersona()}></textarea>
        <//>
        <${Field} label="kategórie" hint="posledná kategória je podpisové odmietnutie tohto hlasu">
          <div class="cr-cats">
            ${categories.map((c, i) => html`<span class="cr-cat" key=${`${i}:${c}`}>
              <input class="cr-input" type="text" value=${c}
                onInput=${(e) => { const next = [...categories]; next[i] = e.target.value; this.setState({ categories: next }); }}
                onChange=${() => this.saveCategories(this.state.categories.filter((x) => x.trim()))} />
              ${i === categories.length - 1 && html`<span class="cr-sig" title="podpisové odmietnutie">podpis</span>`}
              <button class="cr-x" title="odstrániť" onClick=${() => this.saveCategories(categories.filter((_, j) => j !== i))}>x</button>
            </span>`)}
          </div>
          <div class="cr-add">
            <input class="cr-input" type="text" value=${newCat} placeholder="nová kategória"
              onInput=${(e) => this.setState({ newCat: e.target.value })} />
            <button class="cr-btn" disabled=${!newCat.trim()} onClick=${() => {
              this.setState({ newCat: '' });
              this.saveCategories([...categories, newCat.trim()]);
            }}>Pridať</button>
          </div>
        <//>` : null}
      <${Next} ok=${reach(d) >= 4} why="najprv nechaj vygenerovať personu" onClick=${() => this.go(4)} />`;
  }

  renderPhrases(d) {
    const { phrases, categories, perCategory, newLine, busy } = this.state;
    const cats = categories.length ? categories : catsOf(d);
    const total = Object.values(phrases).reduce((n, v) => n + (Array.isArray(v) ? v.length : 0), 0);
    return html`
      <div class="cr-row">
        <${Field} label="viet na kategóriu">
          <input class="cr-input short" type="number" min="1" max="30" value=${perCategory}
            onInput=${(e) => this.setState({ perCategory: e.target.value })} />
        <//>
      </div>
      <div class="cr-row">
        <button class="cr-btn go" disabled=${busy === 'phrases'} onClick=${() => this.writePhrases()}>
          ${busy === 'phrases' ? html`<${Spin} />` : total ? 'Generovať znova' : 'Generovať frázy'}
        </button>
        <span class="cr-hint">${total} viet spolu / prepísanie sa ukladá po opustení poľa</span>
      </div>
      ${cats.map((cat) => {
        const lines = linesOf(phrases, cat);
        return html`<div class="cr-catbox" key=${cat}>
          <h4>${cat}<span class="cr-count ${lines.length < THIN ? 'thin' : ''}">${lines.length}</span></h4>
          ${lines.length < THIN && html`<p class="cr-warn">menej než ${THIN} vety: kategória je tenká, dopíš alebo generuj znova</p>`}
          <div class="cr-lines">${lines.map((text, i) => html`<div class="cr-line" key=${`${cat}:${i}`}>
            <input class="cr-input" type="text" value=${text}
              onInput=${(e) => this.editLine(cat, i, e.target.value)} onChange=${() => this.savePhrases()} />
            <button class="cr-x" title="odstrániť" onClick=${() => this.dropLine(cat, i)}>x</button>
          </div>`)}</div>
          <div class="cr-add">
            <input class="cr-input" type="text" value=${newLine[cat] || ''} placeholder="pridať vetu"
              onInput=${(e) => this.setState((s) => ({ newLine: { ...s.newLine, [cat]: e.target.value } }))} />
            <button class="cr-btn" disabled=${!(newLine[cat] || '').trim()} onClick=${() => this.addLine(cat)}>Pridať</button>
          </div>
        </div>`;
      })}
      <${Next} ok=${total > 0} why="najprv vygeneruj frázy" onClick=${() => this.go(5)} />`;
  }

  renderCommit(d) {
    const { busy, progress, result, commitError, phrases, categories } = this.state;
    if (result) return html`<${GateCard} result=${result} onNew=${() => this.reset()} />`;
    const cats = categories.length ? categories : catsOf(d);
    const total = Object.values(phrases).reduce((n, v) => n + (Array.isArray(v) ? v.length : 0), 0);
    const b = band(d);
    // the id is what every render_id and every path is built from, so a bad one is caught here, not by a 400
    const idErr = idError(d.voice_id);
    const rows = [
      ['meno', d.label], ['id hlasu', d.voice_id || '-'], ['jazyk', d.lang],
      ['klip', `${(num(d.end_s) - num(d.start_s)).toFixed(1)} s z ${d.source_name || 'nahrávky'}`],
      ['výška hlasu', b ? `${b[0]}-${b[1]} Hz` : 'nezmeraná'],
      ['prepis', `${(d.transcript || '').trim().length} znakov`],
      ['soundboard', `${cats.length} kategórií / ${total} viet`],
      ['podpis', cats[cats.length - 1] || '-'],
    ];
    return html`
      <dl class="cr-gate">${rows.map(([k, v]) => html`<div key=${k}><dt>${k}</dt><dd>${v}</dd></div>`)}</dl>
      <p class="cr-hint">Zamkne referenciu, skalibruje bránu na najdlhších vetách, naplní banku a zaradí predrendrovanie.</p>
      ${commitError && html`<div class="banner bad">${commitError}</div>`}
      ${progress && html`<div class="cr-prog">
        <div class="cr-bar"><i style=${`width:${progress.pct}%`}></i></div>
        <span class="cr-hint">${progress.step} - ${progress.pct} %</span>
      </div>`}
      ${idErr && html`<p class="cr-warn">${idErr} - oprav to v kroku 3</p>`}
      <div class="cr-row">
        <button class="speak" disabled=${!!idErr || busy === 'commit' || (!!progress && !commitError)} onClick=${() => this.commit()}>
          ${busy === 'commit' ? html`<${Spin} />` : 'Vytvoriť hlas'}
        </button>
        <button class="cr-btn ghost" onClick=${() => this.discard()}>Zahodiť draft</button>
      </div>`;
  }

  summary(n, d) {
    if (!d) return n === 1 ? 'nahraj nahrávku' : '';
    const b = band(d);
    const cats = this.state.categories.length ? this.state.categories : catsOf(d);
    if (n === 1) return b ? `${(num(d.end_s) - num(d.start_s)).toFixed(1)} s / ${b[0]}-${b[1]} Hz` : d.source_name || 'neorezané';
    if (n === 2) return (d.transcript || '').trim() ? `${d.transcript.trim().slice(0, 60)}...` : 'bez prepisu';
    if (n === 3) return d.voice_id ? `${d.label} / ${d.voice_id} / ${cats.length} kategórií` : 'bez id hlasu';
    if (n === 4) return lineCount(d) ? `${lineCount(d)} viet v ${cats.length} kategóriách` : 'bez fráz';
    return this.state.result ? 'hotovo' : 'pripravené na vytvorenie';
  }

  body(n, d) {
    if (n === 1) return d ? this.renderClip(d) : this.renderUpload();
    if (!d) return null;
    if (n === 2) return this.renderTranscript(d);
    if (n === 3) return this.renderCharacter(d);
    if (n === 4) return this.renderPhrases(d);
    return this.renderCommit(d);
  }

  render(_, s) {
    const d = s.draft;
    const max = reach(d);
    return html`<div class="cr-page">
      <header class="top">
        <span class="brand">Mr. <em>Bag</em></span>
        <span class="vpill">tvorba hlasu</span>
        <span class="dots"><span class="dot ${s.wsUp ? 'on' : ''}"><span class="lbl">link</span></span></span>
        <a class="cr-back" href="/">Späť na Play</a>
      </header>
      ${s.error && html`<div class="banner bad">${s.error}</div>`}
      <main class="cr-main">
        <nav class="cr-steps">${STEPS.map((st) => html`
          <button key=${st.n} class="cr-chip ${st.n === s.step ? 'on' : ''}" disabled=${st.n > max}
            onClick=${() => this.go(st.n)}>${st.n}. ${st.title}</button>`)}</nav>
        ${STEPS.map((st) => html`<${Step} key=${st.n} n=${st.n} title=${st.title} open=${st.n === s.step}
          reached=${st.n <= max} summary=${this.summary(st.n, d)} onOpen=${() => this.go(st.n)}>
          ${this.body(st.n, d)}
        <//>`)}
      </main>
    </div>`;
  }
}

render(html`<${Creator} />`, document.getElementById('app'));
