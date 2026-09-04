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
const BARE = 'bare';                                  // delivery id that adds no token (contract: reads "normalne")
const BARE_LABEL = 'normálne';
const DELIVERY_KEY = 'bag.delivery';
/** Used until GET /api/deliveries answers: the bar must stay usable when the writer router is absent. */
const BARE_ONLY = [{ id: BARE, label: BARE_LABEL, token: '', armed: true, measured: false }];
const LONG_PRESS_MS = 450;   // hold a tile this long and it is edited instead of spoken
const PRESS_SLOP = 10;       // px of drift a press tolerates before it counts as a scroll
const PULSE_MS = 260;        // how long an edited tile keeps its pulse class
const FLASH_MS = 1500;       // how long a freshly saved tile keeps its ring
const CONFIRM_MS = 4000;     // an armed Delete disarms itself: no tile is left one stray tap from gone
/** Where a saved line lands. The server owns the default; this is only how a tile is read back. */
const SAVED_CATEGORY = { sk: 'Moje', en: 'Mine' };
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
  if (!res.ok) {
    const err = new Error(`${method} ${path} -> ${res.status}`);
    err.status = res.status;   // the pencil has to tell 503 (brain absent) from every other failure
    throw err;
  }
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

/** A tile the DM saved himself. The board marks such a line with a source that is not
 *  the bank; while that field is absent the saved category carries the same meaning,
 *  because nothing but a save ever puts a line there. */
const isOwnLine = (line, lang) => (line.source ? line.source !== 'bank' : line.category === SAVED_CATEGORY[lang]);

/* ---------------- long press ---------------- */

/** The one press in flight. It lives outside the component tree on purpose: a press is
 *  a gesture, not a fact about the board, and holding it in state would restart the
 *  timer on every unrelated re-render (during a pre-render, job events land constantly). */
const press = { id: null, timer: null, x: 0, y: 0, fired: false };

function cancelPress() {
  clearTimeout(press.timer);
  press.timer = null;
  press.id = null;
}

/** True once, for the click that follows a fired long press, so the line is not also spoken. */
function tookFired() {
  const was = press.fired;
  press.fired = false;
  return was;
}

/** Pointer props for a tile: a short press plays, a hold edits. Pointer events and not
 *  touch or mouse ones, so a finger, a pen and a mouse all behave the same. */
function pressProps(onTap, onHold) {
  return {
    onPointerDown: (e) => {
      // a second pointer (two fingers) or a non-primary button cancels instead of starting a rival press
      if (press.timer || (e.button != null && e.button > 0)) return cancelPress();
      press.id = e.pointerId;
      press.x = e.clientX;
      press.y = e.clientY;
      press.fired = false;
      press.timer = setTimeout(() => { press.timer = null; press.fired = true; onHold(); }, LONG_PRESS_MS);
    },
    onPointerMove: (e) => {
      if (press.id !== e.pointerId || !press.timer) return;
      if (Math.abs(e.clientX - press.x) > PRESS_SLOP || Math.abs(e.clientY - press.y) > PRESS_SLOP) cancelPress();
    },
    onPointerUp: () => cancelPress(),
    onPointerCancel: () => cancelPress(),
    onContextMenu: (e) => e.preventDefault(),   // a long touch must edit the tile, not open the browser menu
    onClick: (e) => { if (tookFired()) { e.preventDefault(); return; } onTap(); },
  };
}

/* ---------------- delivery selection ---------------- */

/** The delivery is per voice, so one map survives reloads; a private window simply gets bare every time. */
function deliveryMap() {
  try { return JSON.parse(localStorage.getItem(DELIVERY_KEY)) || {}; } catch { return {}; }
}

function storedDelivery(voiceId) {
  const id = deliveryMap()[voiceId];
  return typeof id === 'string' ? id : BARE;
}

function storeDelivery(voiceId, id) {
  try { localStorage.setItem(DELIVERY_KEY, JSON.stringify({ ...deliveryMap(), [voiceId]: id })); } catch { /* ignore */ }
}

/** Bare is always offered and always reads the same, whatever the server calls it. */
const deliveryLabel = (d) => (!d ? BARE_LABEL : d.id === BARE ? BARE_LABEL : d.label);
const selectable = (d) => d.id === BARE || d.armed;
const pickable = (items, id) => (items.some((d) => d.id === id && selectable(d)) ? id : BARE);

/** The brain is usable only while it is resident; anything else keeps the pencil disabled. */
const brainReady = (ready) => ready?.llm === 'resident' || ready?.llm === true;

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

/** The roster ends with the way to a new voice; everything past that link is the Creator's. */
function Roster({ voices, active, onPick }) {
  return html`<nav class="roster">
    ${voices.map((v, i) => html`
    <button key=${v.id} class="voice ${v.id === active ? 'on' : ''}" onClick=${() => onPick(v.id)}>
      <span class="ini">${initials(v.label)}</span>
      <span class="nm">${v.label}<small>${v.lang}${v.locked ? '' : ' / unlocked'}</small></span>
      ${i < 9 && html`<span class="key">Alt+${i + 1}</span>`}
    </button>`)}
    <a class="voice new" href="/creator"><span class="ini">+</span><span class="nm">nový hlas</span></a>
  </nav>`;
}

/** A tile is a button, so its own-tile controls are a sibling inside the cell: a button
 *  nested in a button is invalid and swallows the tap meant for the tile. */
function Tile({ line, job, state, progress, cls = '', keyLabel, own, armed, flash, pulse, onTap, onEdit, onDelete }) {
  const secs = job?.started ? Math.max(0, Math.round((Date.now() - job.started) / 1000)) : 0;
  const badge = state === 'queued' ? (job?.position != null ? `#${job.position}` : 'queued')
    : state === 'rendering' ? `${job?.stage || 'render'} ${secs}s`
    : state === 'error' ? 'tap to retry'
    : state === 'unverified' ? 'unverified'
    : state === 'gate-failed' ? 'gate failed' : null;
  const marks = `${state}${own ? ' owned' : ''}${flash ? ' flash' : ''}${pulse ? ' pulse' : ''}`;
  return html`<div class="cell ${cls}">
    <button class="tile ${cls} ${marks}" title=${line.text} ...${pressProps(onTap, onEdit)}>
      ${keyLabel != null && html`<span class="key">${keyLabel}</span>`}
      <span class="txt">${line.text}</span>
      ${badge && html`<span class="badge">${badge}</span>`}
      ${state === 'rendering' && html`<span class="spin"></span>`}
      ${state === 'playing' && html`<span class="prog" style=${`width:${Math.round(progress * 100)}%`}></span>`}
    </button>
    ${own && html`<span class="own">
      <span class="mine" title="tvoja fráza"></span>
      <button class="del ${armed ? 'armed' : ''}" title=${armed ? 'potvrdiť zmazanie' : 'zmazať frázu'}
        aria-label=${armed ? 'potvrdiť zmazanie' : 'zmazať frázu'} onClick=${onDelete}>${armed ? 'zmazať?' : '×'}</button>
    </span>`}
  </div>`;
}

const EmptyTile = ({ cls, keyLabel, text }) => html`<div class="cell ${cls}">
  <div class="tile ${cls} empty"><span class="key">${keyLabel}</span><span class="txt">${text}</span></div>
</div>`;

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

const Pencil = () => html`<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">
  <path d="M4 20l4.5-1.2L19 8.3l-3.3-3.3L5.2 15.5 4 20z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
  <path d="M14.4 6.3l3.3 3.3" fill="none" stroke="currentColor" stroke-width="1.8" />
</svg>`;

const Star = () => html`<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">
  <path d="M12 3.6l2.6 5.3 5.8.85-4.2 4.1 1 5.75L12 16.9l-5.2 2.7 1-5.75-4.2-4.1 5.8-.85z"
        fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
</svg>`;

const Gear = () => html`<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">
  <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="1.8" />
  <circle cx="12" cy="12" r="7.2" fill="none" stroke="currentColor" stroke-width="1.8" />
  <path d="M12 2.2v2.6M12 19.2v2.6M21.8 12h-2.6M4.8 12H2.2M18.9 5.1l-1.9 1.9M7 17l-1.9 1.9M18.9 18.9L17 17M7 7L5.1 5.1"
        fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" />
</svg>`;

/** The gear opens the delivery menu; the popover shows every spice so the DM can see the mechanism, armed or not. */
function DeliveryPill({ items, value, open, onToggle, onPick }) {
  const current = items.find((d) => d.id === value);
  return html`<div class="delivery">
    <button class="dsel ${value === BARE ? '' : 'on'}" aria-haspopup="menu" aria-expanded=${open ? 'true' : 'false'}
      aria-label="prednes" title=${`prednes: ${deliveryLabel(current)}`} onClick=${onToggle}><${Gear} /></button>
    ${open && html`<div class="dpop" role="menu">${items.map((d) => html`
      <button key=${d.id} class="dopt ${d.id === value ? 'on' : ''}" role="menuitem"
        disabled=${!selectable(d)} onClick=${() => onPick(d.id)}>
        <span class="dlbl">${deliveryLabel(d)}</span>
        ${!selectable(d) && html`<small class="dwhy">neoverené v Labe</small>`}
      </button>`)}</div>`}
  </div>`;
}

/** Returns an array so the ghost line sits under the bar without nesting it inside the flex row. */
function ImprovBar({ text, lang, delivery, deliveries, deliveryOpen, fixing, fixUndo, fixNote, brainDown, boxRef,
                    onText, onLang, onSpeak, onSave, onDelivery, onDeliveryToggle, onFix, onUndo }) {
  const undoKey = (e) => {
    if (!e.ctrlKey || e.key.toLowerCase() !== 'z' || fixUndo === null) return;
    e.preventDefault();                       // the browser's own undo would fight the replacement
    onUndo();
  };
  return [
    html`<div class="improv">
      <input class="line" type="text" maxlength=${LIMIT} autocomplete="off" spellcheck="false"
        placeholder="type a line, Enter speaks" value=${text} readOnly=${fixing} ref=${boxRef}
        onInput=${(e) => onText(e.target.value)} onKeyDown=${undoKey} />
      <button class="fix" title="Opraviť (Ctrl+Enter opraví a povie)" aria-label="opraviť"
        disabled=${fixing || brainDown || !text.trim()} onClick=${onFix}>
        ${fixing ? html`<span class="spin"></span>` : html`<${Pencil} />`}
      </button>
      <button class="save" title="Uložiť ako dlaždicu (Ctrl+S)" aria-label="uložiť"
        disabled=${!text.trim()} onClick=${onSave}><${Star} /></button>
      <span class="count ${text.length > LIMIT - 20 ? 'warn' : ''}">${text.length}/${LIMIT}</span>
      <${DeliveryPill} items=${deliveries} value=${delivery} open=${deliveryOpen}
        onToggle=${onDeliveryToggle} onPick=${onDelivery} />
      <span class="pills">${LANGS.map((l) => html`
        <button key=${l} class="pill ${l === lang ? 'on' : ''}" onClick=${() => onLang(l)}>${l.toUpperCase()}</button>`)}</span>
      <button class="speak" disabled=${!text.trim()} onClick=${onSpeak}>Speak</button>
    </div>`,
    (fixUndo !== null || fixNote) && html`<div class="fixline">
      ${fixUndo !== null
        ? html`<span>opravené — <button class="undo" onClick=${onUndo}>vrátiť</button></span>`
        : html`<span>${fixNote}</span>`}
    </div>`,
  ];
}

function LastTen({ items, meta, onReplay, onRegen, onPin, onSave }) {
  if (!items.length) return html`<div class="last none">the last ${LAST_N} played lines land here: Replay / Regen / Pin / Save</div>`;
  return html`<div class="last">${items.map((it) => {
    const m = meta[it.render_id] || {};
    const dot = m.gate === 'failed' ? 'red' : m.verified === false ? 'amber' : m.gate === 'pass' ? 'green' : '';
    return html`<div class="item" key=${it.key}>
      <span class="ldot ${dot}" title=${m.gate ? `gate ${m.gate}, ${m.verified === false ? 'unverified' : 'verified'}` : ''}></span>
      <span class="lbl" title=${it.label}>${it.label}</span>
      <button onClick=${() => onReplay(it)}>Replay</button>
      <button onClick=${() => onRegen(it)} title="take +1">Regen</button>
      <button class=${m.pinned ? 'on' : ''} disabled=${!m.line_id} onClick=${() => onPin(it)} title="Ctrl+P pins the last one">${m.pinned ? 'Pinned' : 'Pin'}</button>
      <button class="star" onClick=${() => onSave(it)} aria-label="uložiť ako dlaždicu" title="Uložiť ako dlaždicu"><${Star} /></button>
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
    // improv bar: the chosen delivery for the active voice, and the pencil's one-shot undo
    deliveries: BARE_ONLY, delivery: BARE, deliveryOpen: false,
    fixing: false, fixUndo: null, fixNote: null, brainDown: false,
    // one-shot tile marks: the saved tile's flash, the edited tile's pulse, the armed Delete
    flash: null, pulse: null, confirmDelete: null,
  };
  id = clientId();
  box = null;            // the improv input, so a long-pressed tile can put the caret in it

  componentDidMount() {
    armAudio(audio);
    audio.addEventListener('timeupdate', () => this.setState({ progress: audio.duration ? audio.currentTime / audio.duration : 0 }));
    audio.addEventListener('ended', () => this.onEnded());
    window.addEventListener('keydown', (e) => this.onKey(e));
    // pointerdown, not click: the popover must be gone before the press lands anywhere else
    window.addEventListener('pointerdown', (e) => {
      if (this.state.deliveryOpen && !e.target.closest?.('.delivery')) this.setState({ deliveryOpen: false });
      if (this.state.confirmDelete && !e.target.closest?.('.own')) this.setState({ confirmDelete: null });
    });
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
      // The Creator hands the DM its finished voice as /?voice=<id>. It can only win
      // on the very first boot: from the second one on `kept` holds whatever voice is
      // actually on screen, so a reconnect never yanks the table back to that link.
      const asked = voices.find((v) => v.id === new URLSearchParams(location.search).get('voice'));
      const voice = kept || asked || voices.find((v) => v.id === 'bag') || voices.find((v) => v.locked) || voices[0] || null;
      this.setState({ voices, voice, lang: kept ? this.state.lang : voice?.lang || 'sk', error: null }, () => this.loadBoard());
      this.loadDeliveries(voice, true);
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

  /** The spice list is per voice. `restore` keeps the stored choice (boot, reconnect); a voice change resets to bare.
   *  A missing endpoint is not something the DM can act on: the bar falls back to bare and keeps working. */
  async loadDeliveries(voice, restore) {
    if (!voice) return this.setState({ deliveries: BARE_ONLY, delivery: BARE, deliveryOpen: false });
    let items;
    try { items = await api('GET', `/api/deliveries?voice=${encodeURIComponent(voice.id)}`); }
    catch { items = BARE_ONLY; }
    if (!Array.isArray(items) || !items.length) items = BARE_ONLY;
    const delivery = pickable(items, restore ? storedDelivery(voice.id) : BARE);
    storeDelivery(voice.id, delivery);
    this.setState((st) => (st.voice?.id === voice.id ? { deliveries: items, delivery, deliveryOpen: false } : null));
  }

  setDelivery(id) {
    const voice = this.state.voice;
    if (!voice) return;
    const delivery = pickable(this.state.deliveries, id);   // an unarmed spice is never selectable
    storeDelivery(voice.id, delivery);
    this.setState({ delivery, deliveryOpen: false });
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
    // the pencil re-arms only when a status event says the brain is resident again
    this.setState({ ready, player, queueDepth, ...(brainReady(ready) ? { brainDown: false } : {}) });
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

  /** `delivery` is sent only when the improv bar asked for a spice: board tiles stay bare by contract.
   *  `line_id` travels with every render that belongs to a tile, because that is what makes the server
   *  adopt the render for the line; without it the tile lights up locally and goes grey on the next board read. */
  async say({ text, lineId, label, takeNo = 0, delivery = null }) {
    const { voice, lang } = this.state;
    if (!voice || !text.trim()) return;
    const body = { voice: voice.id, text, lang, priority: 'live', take_no: takeNo };
    if (lineId) body.line_id = lineId;
    if (delivery && delivery !== BARE) body.delivery = delivery;
    try {
      const r = await api('POST', '/api/say', body);
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

  /** Long press, or Shift and the tile's key: the line lands in the improv box instead of
   *  being spoken, so a bank line can be bent to what is actually happening at the table. */
  editLine(line) {
    if (!line) return;
    clearTimeout(this.pulseTimer);
    this.pulseTimer = setTimeout(() => this.setState({ pulse: null }), PULSE_MS);
    this.setState({ text: line.text, fixUndo: null, fixNote: null, pulse: line.id }, () => {
      const box = this.box;
      if (!box) return;
      box.focus();
      box.setSelectionRange(box.value.length, box.value.length);   // caret at the end: the DM edits the tail
    });
  }

  /** The star: the text becomes a tile of the DM's own. The server picks the saved category and
   *  returns the line that already exists when the same text is saved twice, so this cannot
   *  make a duplicate tile; the board is re-read because the category itself may be new. */
  async saveLine(text) {
    const { voice, lang } = this.state;
    const clean = (text || '').trim();
    if (!voice || !clean) return;
    try {
      const line = await api('POST', '/api/lines', { voice: voice.id, lang, text: clean });
      clearTimeout(this.flashTimer);
      this.flashTimer = setTimeout(() => this.setState({ flash: null }), FLASH_MS);
      this.setState({ tab: line.category, flash: line.id });
      await this.loadBoard();
    } catch (e) { this.fail(e); }
  }

  /** Two taps, because a line the DM wrote is gone for good. The armed state expires on its
   *  own and on the next press elsewhere, so no tile is left one stray tap from deletion. */
  askDelete(line) {
    clearTimeout(this.confirmTimer);
    if (this.state.confirmDelete !== line.id) {
      this.confirmTimer = setTimeout(() => this.setState({ confirmDelete: null }), CONFIRM_MS);
      this.setState({ confirmDelete: line.id });
      return;
    }
    this.setState({ confirmDelete: null });
    api('DELETE', `/api/lines/${line.id}`).then(() => this.loadBoard()).catch((e) => this.fail(e));
  }

  /** `override` lets Ctrl+Enter speak the text the fix just produced without waiting for a state flush. */
  speak(override) {
    const text = (override ?? this.state.text).trim();
    if (!text) return;
    this.setState({ text: '', fixUndo: null, fixNote: null });
    this.say({ text, lineId: null, label: text, delivery: this.state.delivery });
  }

  /** The pencil: one pass of the Writer over the box. Returns the text now in the box, null if it never answered. */
  async fix() {
    const { voice, lang, text, fixing, brainDown } = this.state;
    const original = text.trim();
    if (!voice || !original || fixing || brainDown) return null;
    this.setState({ fixing: true, fixNote: null });
    try {
      const r = await api('POST', '/api/fix', { voice: voice.id, text: original, lang });
      // the undo keeps the raw box content, not the trimmed line that was sent, so it restores exactly
      if (r.changed) this.setState({ text: r.text, fixUndo: text, fixNote: null });
      else this.setState({ fixNote: r.note || 'bez zmeny', fixUndo: null });
      return r.changed ? r.text : text;
    } catch (e) {
      if (e.status === 503) this.setState({ brainDown: true, fixNote: null });
      else this.fail(e);
      return null;
    } finally {
      this.setState({ fixing: false });
    }
  }

  /** Ctrl+Enter. A brain that never answered leaves the line in the box, so Enter still speaks it as typed. */
  async fixThenSpeak() {
    const fixed = await this.fix();
    if (fixed !== null) this.speak(fixed);
  }

  /** Restores the pre-fix text character for character; the ghost line and Ctrl+Z share it. */
  undoFix() {
    this.setState((s) => (s.fixUndo === null ? null : { text: s.fixUndo, fixUndo: null, fixNote: null }));
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
    this.setState({ voice, lang: voice.lang || this.state.lang, board: null, tab: null,
      deliveries: BARE_ONLY, delivery: BARE, deliveryOpen: false }, () => this.loadBoard());
    this.loadDeliveries(voice, false);
  }

  setLang(lang) {
    if (lang === this.state.lang) return;
    this.setState({ lang, board: null }, () => this.loadBoard());
  }

  /** Keyboard map from the contract; digits and letters are ignored while typing, Enter, Esc
   *  and Ctrl+S never are (Ctrl+S saves the line being typed, so it has to reach the box).
   *  Voice switching moved to Alt+1..9 because Shift and a tile's key now edits that tile. */
  onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); return this.stop(); }
    if (e.key === 'Enter') { e.preventDefault(); return e.ctrlKey ? this.fixThenSpeak() : this.speak(); }
    if (e.ctrlKey && e.key.toLowerCase() === 's') { e.preventDefault(); return this.saveLine(this.state.text); }
    if (isTyping(e.target) || e.metaKey) return;
    if (e.altKey) {
      const n = /^Digit([1-9])$/.exec(e.code);
      if (n && this.state.voices[n[1] - 1]) { e.preventDefault(); this.pickVoice(this.state.voices[n[1] - 1].id); }
      return;
    }
    if (e.ctrlKey) {
      if (e.key.toLowerCase() === 'p') { e.preventDefault(); this.pin(this.state.last[0] || {}); }
      return;
    }
    if (e.shiftKey) {
      const n = /^Digit([1-8])$/.exec(e.code);
      if (n) { e.preventDefault(); return this.editLine(favLine(this.state.board, +n[1])); }
      if (e.code === 'KeyT') { e.preventDefault(); return this.editLine(tenLine(this.state.board)); }
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
      progress=${s.progress} own=${isOwnLine(line, s.lang)} armed=${s.confirmDelete === line.id}
      flash=${s.flash === line.id} pulse=${s.pulse === line.id}
      onTap=${() => this.tapLine(line)} onEdit=${() => this.editLine(line)}
      onDelete=${() => this.askDelete(line)} ...${extra} />`;
    const site = bannerFor(s.wsUp, s.ready, s.player);
    // a dead brain outranks the standing warnings (a DM playing locally always has 'no speaker'),
    // but never the two 'bad' ones: a lost socket or a dead TTS is the bigger problem on the table
    const banner = s.error ? ['bad', s.error]
      : site?.[0] === 'bad' ? site
      : s.brainDown ? ['warn', 'mozog nie je pripravený'] : site;
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
        <${ImprovBar} text=${s.text} lang=${s.lang} delivery=${s.delivery} deliveries=${s.deliveries}
          deliveryOpen=${s.deliveryOpen} fixing=${s.fixing} fixUndo=${s.fixUndo} fixNote=${s.fixNote}
          brainDown=${s.brainDown} boxRef=${(el) => { this.box = el; }}
          onText=${(text) => this.setState({ text })}
          onLang=${(l) => this.setLang(l)} onSpeak=${() => this.speak()} onSave=${() => this.saveLine(s.text)}
          onDelivery=${(id) => this.setDelivery(id)}
          onDeliveryToggle=${() => this.setState((st) => ({ deliveryOpen: !st.deliveryOpen }))}
          onFix=${() => this.fix()} onUndo=${() => this.undoFix()} />
        <${LastTen} items=${s.last} meta=${s.meta} onReplay=${(it) => this.play(it.render_id, it.label)}
          onRegen=${(it) => this.regenerate(it)} onPin=${(it) => this.pin(it)}
          onSave=${(it) => this.saveLine(s.meta[it.render_id]?.text || it.label)} />
      </footer>
    </div>`;
  }
}

render(html`<${App} />`, document.getElementById('app'));
