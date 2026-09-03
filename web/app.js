/**
 * Play page: the table console. One Preact tree, one WebSocket, one <audio>.
 *
 * All state lives in the App class and every other component is a pure
 * function of props: the vendored preact core has no hooks module, and one
 * owner of state keeps the event -> tile-state mapping in a single place.
 * Server events are the truth; local patches only bridge the gap until the
 * next board refresh so a tap feels instant.
 */
import { h, render, Component } from './vendor/preact.min.js';
import htm from './vendor/htm.js';

const html = htm.bind(h);
const LIMIT = 250;        // improv bar hard cap (contract: "250 counter")
const LAST_N = 10;
const SLOTS = 8;
const LANGS = ['sk', 'en'];
const audio = document.getElementById('audio');

/* ---------------- helpers ---------------- */

/** Stable per-browser id so the hub can tell tablets apart across reloads; per role so one tablet can run both pages. */
function clientId() {
  const key = 'bag.client.play';
  let id = null;
  try { id = localStorage.getItem(key); } catch { /* private mode: a per-load id is fine */ }
  if (!id) {
    id = Math.random().toString(36).slice(2, 10);
    try { localStorage.setItem(key, id); } catch { /* ignore */ }
  }
  return id;
}

const wsUrl = (id) => `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?client=${id}&role=play`;

/** JSON fetch that throws on HTTP errors so every caller surfaces them the same way. */
async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}`);
  return res.status === 204 ? null : res.json();
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
  connect();
  return { send: (obj) => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); } };
}

/** Tiny silent WAV: playing it inside the first gesture unlocks the element for later, event-driven playback. */
function silentWav() {
  const n = 800;
  const buf = new ArrayBuffer(44 + n);
  const v = new DataView(buf);
  const tag = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  tag(0, 'RIFF'); v.setUint32(4, 36 + n, true); tag(8, 'WAVE'); tag(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 8000, true); v.setUint32(28, 8000, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true);
  tag(36, 'data'); v.setUint32(40, n, true);
  new Uint8Array(buf, 44).fill(128);
  return URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
}

/** Autoplay policy: one play() inside a user gesture, after which play.start events may drive the element. */
function armAudio(el) {
  const unlock = () => {
    window.removeEventListener('pointerdown', unlock);
    window.removeEventListener('keydown', unlock);
    if (!el.paused) return;
    if (!el.src) el.src = silentWav();       // nothing pending: play silence; otherwise the blocked line plays now
    const src = el.src;
    el.play().then(() => { if (el.src === src && src.startsWith('blob:')) el.pause(); }).catch(() => {});
  };
  window.addEventListener('pointerdown', unlock);
  window.addEventListener('keydown', unlock);
}

const isTyping = (el) => !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
const initials = (label) => label.split(/\s+/).slice(0, 2).map((w) => w[0] || '').join('').toUpperCase();

/** Server status overlaid with what this client saw in job and play events. */
function tileState(line, job, now) {
  const rid = job?.render_id || line.render_id;
  if (now && rid && now.render_id === rid) return 'playing';
  if (job?.status === 'queued') return 'queued';
  if (job?.status === 'running') return 'rendering';
  if (job?.status === 'failed') return 'error';
  return line.status || 'pending';
}

const statusOf = (done) => (done.gate === 'failed' ? 'gate-failed' : done.verified === false ? 'unverified' : 'ready');

/** Slot -> line: the board's favourites list first, the per-line slot column as fallback. */
function favLine(board, slot) {
  if (!board) return null;
  const f = board.favourites;
  const id = Array.isArray(f) ? f[slot - 1] : f?.[slot];
  return board.lines.find((l) => l.id === id) || board.lines.find((l) => l.slot === slot) || null;
}
const tenLine = (board) => board?.lines.find((l) => l.id === board.ten_nie) || null;

const patchLine = (board, id, patch) => ({ ...board, lines: board.lines.map((l) => (l.id === id ? { ...l, ...patch } : l)) });

/** Forget older jobs of a line when a new one is submitted so a failed job cannot shadow its retry. */
function dropLine(jobs, lineId) {
  if (!lineId) return jobs;
  return Object.fromEntries(Object.entries(jobs).filter(([, j]) => j.line_id !== lineId));
}

/** One banner at a time, worst problem first. */
function bannerFor(wsUp, ready, player) {
  if (!wsUp) return ['bad', 'connection lost: reconnecting'];
  if (ready && ready.tts === false) return ['bad', 'TTS down: cached lines only'];
  if (player && !player.speaker) return ['warn', 'no speaker connected (playing here)'];
  if (ready && ready.stt === false) return ['warn', 'whisper down: lines unverified'];
  return null;
}

/* ---------------- pure components ---------------- */

const Dot = ({ label, state }) => html`<span class="dot ${state}" title=${label}><span class="lbl">${label}</span></span>`;

function TopBar({ voice, ready, speaker, queued, pending, onStop }) {
  const tts = !ready?.tts ? '' : ready.tts_warm === false ? 'warm' : 'on';
  const llm = ready?.llm === 'resident' || ready?.llm === true ? 'on' : '';
  return html`<header class="top">
    <span class="brand">Mr. <em>Bag</em></span>
    ${voice && html`<span class="vpill">${voice.label}</span>`}
    <span class="dots">
      <${Dot} label="tts" state=${tts} />
      <${Dot} label="whisper" state=${ready?.stt ? 'on' : ''} />
      <${Dot} label="brain" state=${llm} />
      <${Dot} label="speaker" state=${speaker ? 'on' : ''} />
    </span>
    <span class="qcount">render ${queued}${pending ? ` / play ${pending}` : ''}</span>
    <button class="stop" onClick=${onStop} title="Stop (Esc)">STOP</button>
  </header>`;
}

function VoiceCard({ voice }) {
  if (!voice) return null;
  const flags = [voice.lang, `v${voice.version}`, `energy ${voice.energy}`, voice.locked ? 'locked' : 'unlocked'];
  if (voice.calibrated === false) flags.push('not calibrated');
  return html`<section class="card"><h2>${voice.label}</h2><div class="meta">${flags.join(' / ')}</div></section>`;
}

function Roster({ voices, active, onPick }) {
  return html`<nav class="roster">${voices.map((v, i) => html`
    <button key=${v.id} class="voice ${v.id === active ? 'on' : ''}" onClick=${() => onPick(v.id)}>
      <span class="ini">${initials(v.label)}</span>
      <span class="nm">${v.label}<small>${v.lang}${v.locked ? '' : ' / unlocked'}</small></span>
      ${i < 9 && html`<span class="key">Shift+${i + 1}</span>`}
    </button>`)}</nav>`;
}

function Tile({ line, job, state, progress, cls = '', keyLabel, onTap }) {
  const secs = job?.started ? Math.max(0, Math.round((Date.now() - job.started) / 1000)) : 0;
  const badge = state === 'queued' ? (job?.position != null ? `#${job.position}` : 'queued')
    : state === 'rendering' ? `${job?.stage || 'render'} ${secs}s`
    : state === 'error' ? 'tap to retry'
    : state === 'unverified' ? 'unverified'
    : state === 'gate-failed' ? 'gate failed' : null;
  return html`<button class="tile ${cls} ${state}" onClick=${onTap} title=${line.text}>
    ${keyLabel != null && html`<span class="key">${keyLabel}</span>`}
    <span class="txt">${line.text}</span>
    ${badge && html`<span class="badge">${badge}</span>`}
    ${state === 'rendering' && html`<span class="spin"></span>`}
    ${state === 'playing' && html`<span class="prog" style=${`width:${Math.round(progress * 100)}%`}></span>`}
  </button>`;
}

const EmptyTile = ({ cls, keyLabel, text }) => html`<div class="tile ${cls} empty"><span class="key">${keyLabel}</span><span class="txt">${text}</span></div>`;

function Favourites({ board, tile }) {
  const ten = tenLine(board);
  return html`<section class="favs">
    ${Array.from({ length: SLOTS }, (_, i) => {
      const line = favLine(board, i + 1);
      return line ? tile(line, { key: line.id, cls: 'fav', keyLabel: i + 1 }) : html`<${EmptyTile} key=${'e' + i} cls="fav" keyLabel=${i + 1} text="-" />`;
    })}
    ${ten ? tile(ten, { key: ten.id, cls: 'fav tennie', keyLabel: 'T' }) : html`<${EmptyTile} cls="fav tennie" keyLabel="T" text="Ten nie." />`}
  </section>`;
}

const Tabs = ({ cats, active, onPick }) => html`<nav class="tabs">${cats.map((c) => html`
  <button key=${c} class="tab ${c === active ? 'on' : ''}" onClick=${() => onPick(c)}>${c}</button>`)}</nav>`;

function Grid({ lines, loading, tile }) {
  if (loading) return html`<div class="grid note">loading board</div>`;
  if (!lines.length) return html`<div class="grid note">no lines in this category</div>`;
  return html`<section class="grid">${lines.map((l) => tile(l, { key: l.id }))}</section>`;
}

function ImprovBar({ text, lang, onText, onLang, onSpeak }) {
  return html`<div class="improv">
    <input class="line" type="text" maxlength=${LIMIT} autocomplete="off" spellcheck="false"
      placeholder="type a line, Enter speaks" value=${text} onInput=${(e) => onText(e.target.value)} />
    <span class="count ${text.length > LIMIT - 20 ? 'warn' : ''}">${text.length}/${LIMIT}</span>
    <span class="pills">${LANGS.map((l) => html`
      <button key=${l} class="pill ${l === lang ? 'on' : ''}" onClick=${() => onLang(l)}>${l.toUpperCase()}</button>`)}</span>
    <button class="speak" disabled=${!text.trim()} onClick=${onSpeak}>Speak</button>
  </div>`;
}

function LastTen({ items, meta, onReplay, onRegen, onPin }) {
  if (!items.length) return html`<div class="last none">the last ${LAST_N} played lines land here: Replay / Regen / Pin</div>`;
  return html`<div class="last">${items.map((it) => {
    const m = meta[it.render_id] || {};
    const dot = m.gate === 'failed' ? 'red' : m.verified === false ? 'amber' : m.gate === 'pass' ? 'green' : '';
    return html`<div class="item" key=${it.key}>
      <span class="ldot ${dot}" title=${m.gate ? `gate ${m.gate}, ${m.verified === false ? 'unverified' : 'verified'}` : ''}></span>
      <span class="lbl" title=${it.label}>${it.label}</span>
      <button onClick=${() => onReplay(it)}>Replay</button>
      <button onClick=${() => onRegen(it)} title="take +1">Regen</button>
      <button class=${m.pinned ? 'on' : ''} disabled=${!m.line_id} onClick=${() => onPin(it)} title="Ctrl+P pins the last one">${m.pinned ? 'Pinned' : 'Pin'}</button>
    </div>`;
  })}</div>`;
}

/* ---------------- App: state, socket, actions ---------------- */

class App extends Component {
  state = {
    voices: [], voice: null, lang: 'sk', board: null, tab: null,
    ready: null, wsUp: false, player: null, queueDepth: null,
    jobs: {},     // job_id -> {id, line_id, label, text, take_no, status, position, started, stage, autoplay, render_id}
    meta: {},     // render_id -> {line_id, text, take_no, verified, gate, pinned}
    last: [],     // newest first: {key, render_id, label}
    text: '', progress: 0, error: null,
  };
  id = clientId();

  componentDidMount() {
    armAudio(audio);
    audio.addEventListener('timeupdate', () => this.setState({ progress: audio.duration ? audio.currentTime / audio.duration : 0 }));
    audio.addEventListener('ended', () => this.onEnded());
    window.addEventListener('keydown', (e) => this.onKey(e));
    this.sock = openSocket(wsUrl(this.id), {
      open: () => { this.setState({ wsUp: true }); this.boot(); },
      close: () => this.setState({ wsUp: false }),
      event: (ev, d) => this.onEvent(ev, d),
    });
  }

  /** Rendering tiles show elapsed seconds; tick only while something is rendering. */
  componentDidUpdate() {
    const running = Object.values(this.state.jobs).some((j) => j.status === 'running');
    if (running && !this.ticker) this.ticker = setInterval(() => this.forceUpdate(), 1000);
    if (!running && this.ticker) { clearInterval(this.ticker); this.ticker = null; }
  }

  /** Runs on every (re)connect: voices, player and board may have drifted while we were away. */
  async boot() {
    try {
      const voices = await api('GET', '/api/voices');
      const kept = this.state.voice && voices.find((v) => v.id === this.state.voice.id);
      const voice = kept || voices.find((v) => v.id === 'bag') || voices.find((v) => v.locked) || voices[0] || null;
      this.setState({ voices, voice, lang: kept ? this.state.lang : voice?.lang || 'sk', error: null }, () => this.loadBoard());
      this.setState({ player: await api('GET', '/api/player') });
    } catch (e) { this.fail(e); }
  }

  async loadBoard() {
    const { voice, lang } = this.state;
    if (!voice) return;
    try {
      const board = await api('GET', `/api/board?voice=${encodeURIComponent(voice.id)}&lang=${lang}`);
      this.setState((s) => (s.voice?.id === voice.id && s.lang === lang
        ? { board, tab: board.categories.includes(s.tab) ? s.tab : board.categories[0] || null }
        : null));
    } catch (e) { this.fail(e); }
  }

  /** Board refreshes are debounced: boot pre-render finishes dozens of jobs per minute. */
  refreshSoon() {
    clearTimeout(this.refreshTimer);
    this.refreshTimer = setTimeout(() => this.loadBoard(), 600);
  }

  fail(e) {
    this.setState({ error: e.message });
    clearTimeout(this.errTimer);
    this.errTimer = setTimeout(() => this.setState({ error: null }), 4000);
  }

  /* ---- socket events ---- */

  onEvent(ev, d) {
    switch (ev) {
      case 'status': return this.applyStatus(d);
      case 'job.queued': return this.trackJob(d, 'queued');
      case 'job.started': return this.trackJob(d, 'running', { started: Date.now() });
      case 'job.progress': return this.trackJob(d, 'running', { stage: d.stage });
      case 'job.done': return this.onDone(d);
      case 'job.failed': return this.trackJob(d, 'failed');
      case 'job.cancelled': return this.forgetJob(d.job_id ?? d.id);
      case 'queue.changed': return this.setState({ queueDepth: d.depth ?? 0 });
      case 'play.start': return this.onPlayStart(d);
      case 'play.end': return this.onPlayEnd(false);
      case 'play.stop': return this.onPlayEnd(true);
      case 'speaker.presence': return this.patchPlayer({ speaker: !!d.connected });
      default: return undefined;
    }
  }

  applyStatus(d) {
    const ready = d.ready ?? d.readyz ?? d;
    const player = d.player ?? (d.now !== undefined ? d : this.state.player);
    const queueDepth = Array.isArray(d.queue) ? d.queue.length : this.state.queueDepth;
    this.setState({ ready, player, queueDepth });
  }

  /** A cancelled job leaves no trace: the tile falls back to whatever the server says about the line. */
  forgetJob(id) {
    this.setState((s) => {
      const jobs = { ...s.jobs };
      delete jobs[id];
      return { jobs };
    });
  }

  patchPlayer(patch) { this.setState((s) => ({ player: { ...(s.player || {}), ...patch } })); }

  /** Merge a job event; jobs we did not submit are tracked only when the event names their line.
   *  `position` is the worker's number at enqueue time: queue.changed carries only depth, so it is not re-numbered. */
  trackJob(d, status, extra = {}) {
    const id = d.job_id ?? d.id;
    if (!id) return;
    this.setState((s) => {
      const prev = s.jobs[id];
      if (!prev && !d.line_id) return null;
      const seen = { line_id: d.line_id ?? prev?.line_id ?? null, render_id: d.render_id ?? prev?.render_id, position: d.position ?? prev?.position };
      const job = { id, autoplay: false, take_no: d.take_no ?? 0, text: d.text, ...prev, ...seen, ...extra, status };
      return { jobs: { ...(prev ? s.jobs : dropLine(s.jobs, d.line_id)), [id]: job } };
    });
  }

  onDone(d) {
    const id = d.job_id ?? d.id;
    const job = this.state.jobs[id];
    const rid = d.render_id;
    this.setState((s) => {
      const jobs = { ...s.jobs };
      delete jobs[id];
      const lineId = job?.line_id ?? d.line_id ?? s.meta[rid]?.line_id ?? null;
      const meta = { ...s.meta, [rid]: { ...s.meta[rid], line_id: lineId, text: job?.text, take_no: job?.take_no, verified: d.verified, gate: d.gate } };
      const board = lineId && s.board ? patchLine(s.board, lineId, { render_id: rid, status: statusOf(d) }) : s.board;
      return { jobs, meta, board };
    });
    if (job?.autoplay) this.play(rid, job.label);      // a gate-failed take still plays: the strip's red dot invites Regen
    this.refreshSoon();
  }

  onPlayStart(d) {
    const local = !this.state.player?.speaker;
    this.setState((s) => ({
      player: { ...(s.player || {}), now: d },
      last: [{ key: `${d.render_id}:${Date.now()}`, render_id: d.render_id, label: d.label || s.meta[d.render_id]?.text || d.render_id }, ...s.last].slice(0, LAST_N),
      progress: 0,
    }));
    if (!this.state.meta[d.render_id]?.gate) this.fetchMeta(d.render_id);
    if (!local) return;
    audio.src = d.url;
    audio.play().catch(() => this.fail(new Error('tap anywhere once to allow audio')));
  }

  onPlayEnd(stopped) {
    if (stopped) { audio.pause(); audio.removeAttribute('src'); audio.load(); }
    this.setState((s) => ({ player: { ...(s.player || {}), now: null }, progress: 0 }));
  }

  /** The page that actually played the line reports its end so the server FIFO advances. */
  onEnded() {
    const rid = this.state.player?.now?.render_id;
    if (rid) this.sock.send({ type: 'ended', render_id: rid });
  }

  /** Renders played from elsewhere (remote, cached tiles) need their row for dots, line id and take number. */
  fetchMeta(rid) {
    api('GET', `/api/renders/${rid}`)
      .then((row) => this.noteMeta(rid, { line_id: this.state.meta[rid]?.line_id ?? row.line_id, take_no: row.take_no, verified: !!row.verified, gate: row.gate }))
      .catch(() => {});
  }

  noteMeta(rid, patch) { this.setState((s) => ({ meta: { ...s.meta, [rid]: { ...s.meta[rid], ...patch } } })); }

  registerJob(id, fields) {
    this.setState((s) => ({ jobs: { ...dropLine(s.jobs, fields.line_id), [id]: { status: 'queued', ...s.jobs[id], ...fields, id } } }));
  }

  /* ---- actions ---- */

  async say({ text, lineId, label, takeNo = 0 }) {
    const { voice, lang } = this.state;
    if (!voice || !text.trim()) return;
    try {
      const r = await api('POST', '/api/say', { voice: voice.id, text, lang, priority: 'live', take_no: takeNo });
      this.noteMeta(r.render_id, { line_id: lineId, text, take_no: takeNo });
      if (r.cached) return this.play(r.render_id, label);
      this.registerJob(r.job_id, { line_id: lineId, label, text, take_no: takeNo, position: r.position, autoplay: true });
    } catch (e) { this.fail(e); }
  }

  async play(renderId, label) {
    try { await api('POST', '/api/play', { render_id: renderId, label }); } catch (e) { this.fail(e); }
  }

  /** Tile tap: ready plays now; anything else asks the worker (cache makes that instant when a pass take exists). */
  tapLine(line) {
    if (!line) return;
    const st = tileState(line, this.jobByLine()[line.id], this.state.player?.now);
    if (st === 'queued' || st === 'rendering') return;
    if ((st === 'ready' || st === 'playing') && line.render_id) {
      this.noteMeta(line.render_id, { line_id: line.id, text: line.text });
      return this.play(line.render_id, line.text);
    }
    this.say({ text: line.text, lineId: line.id, label: line.text });
  }

  speak() {
    const text = this.state.text.trim();
    if (!text) return;
    this.setState({ text: '' });
    this.say({ text, lineId: null, label: text });
  }

  async regenerate(it) {
    const m = this.state.meta[it.render_id] || {};
    const takeNo = (m.take_no ?? 0) + 1;
    if (!m.line_id) return this.say({ text: m.text || it.label, lineId: null, label: it.label, takeNo });
    try {
      const r = await api('POST', `/api/lines/${m.line_id}/regenerate`, {});
      this.registerJob(r.job_id, { line_id: m.line_id, label: it.label, text: m.text, take_no: takeNo, autoplay: true });
    } catch (e) { this.fail(e); }
  }

  async pin(it) {
    const lineId = this.state.meta[it.render_id]?.line_id;
    if (!lineId) return;
    try {
      await api('POST', `/api/renders/${it.render_id}/pin`, { line_id: lineId });
      this.noteMeta(it.render_id, { pinned: true });
      this.refreshSoon();
    } catch (e) { this.fail(e); }
  }

  stop() {
    audio.pause();                                   // local first: Esc must feel instant, the event confirms
    api('POST', '/api/stop').catch((e) => this.fail(e));
  }

  pickVoice(id) {
    const voice = this.state.voices.find((v) => v.id === id);
    if (!voice || voice.id === this.state.voice?.id) return;
    this.setState({ voice, lang: voice.lang || this.state.lang, board: null, tab: null }, () => this.loadBoard());
  }

  setLang(lang) {
    if (lang === this.state.lang) return;
    this.setState({ lang, board: null }, () => this.loadBoard());
  }

  /** Keyboard map from the contract; digits and letters are ignored while typing, Enter and Esc never are. */
  onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); return this.stop(); }
    if (e.key === 'Enter') { e.preventDefault(); return this.speak(); }
    if (isTyping(e.target) || e.altKey || e.metaKey) return;
    if (e.ctrlKey) {
      if (e.key.toLowerCase() === 'p') { e.preventDefault(); this.pin(this.state.last[0] || {}); }
      return;
    }
    if (e.shiftKey) {
      const n = /^Digit([1-9])$/.exec(e.code);
      if (n && this.state.voices[n[1] - 1]) { e.preventDefault(); this.pickVoice(this.state.voices[n[1] - 1].id); }
      return;
    }
    const k = e.key.toLowerCase();
    const action = /^[1-8]$/.test(k) ? () => this.tapLine(favLine(this.state.board, +k))
      : k === 't' ? () => this.tapLine(tenLine(this.state.board))
      : k === 'r' ? () => api('POST', '/api/repeat').catch((err) => this.fail(err))
      : k === ' ' ? () => api('POST', '/api/next').catch((err) => this.fail(err))
      : null;
    if (action) { e.preventDefault(); action(); }
  }

  jobByLine() {
    const out = {};
    for (const j of Object.values(this.state.jobs)) if (j.line_id) out[j.line_id] = j;
    return out;
  }

  render(_, s) {
    const byLine = this.jobByLine();
    const now = s.player?.now || null;
    const tile = (line, extra) => html`<${Tile} line=${line} job=${byLine[line.id]} state=${tileState(line, byLine[line.id], now)}
      progress=${s.progress} onTap=${() => this.tapLine(line)} ...${extra} />`;
    const banner = s.error ? ['bad', s.error] : bannerFor(s.wsUp, s.ready, s.player);
    const catLines = (s.board?.lines || []).filter((l) => l.category === s.tab);
    return html`<div class="play">
      <${TopBar} voice=${s.voice} ready=${s.ready} speaker=${!!s.player?.speaker}
        queued=${s.queueDepth ?? s.ready?.queue_depth ?? 0} pending=${s.player?.queue?.length || 0}
        onStop=${() => this.stop()} />
      ${banner && html`<div class="banner ${banner[0]}">${banner[1]}</div>`}
      <div class="main">
        <aside class="left">
          <${VoiceCard} voice=${s.voice} />
          <${Roster} voices=${s.voices} active=${s.voice?.id} onPick=${(id) => this.pickVoice(id)} />
        </aside>
        <div class="centre">
          <${Favourites} board=${s.board} tile=${tile} />
          <${Tabs} cats=${s.board?.categories || []} active=${s.tab} onPick=${(tab) => this.setState({ tab })} />
          <${Grid} lines=${catLines} loading=${!s.board} tile=${tile} />
        </div>
      </div>
      <footer class="bottom">
        <${ImprovBar} text=${s.text} lang=${s.lang} onText=${(text) => this.setState({ text })}
          onLang=${(l) => this.setLang(l)} onSpeak=${() => this.speak()} />
        <${LastTen} items=${s.last} meta=${s.meta} onReplay=${(it) => this.play(it.render_id, it.label)}
          onRegen=${(it) => this.regenerate(it)} onPin=${(it) => this.pin(it)} />
      </footer>
    </div>`;
  }
}

render(html`<${App} />`, document.getElementById('app'));
