/**
 * Speaker page: the one client that owns the room's audio.
 *
 * A single tap on "Claim speaker" is the user gesture that unlocks autoplay;
 * from then on every play.start from the hub drives the <audio> element and
 * the page reports "ended" so the server FIFO can advance. State stays on the
 * server: this page only executes it, which is why a locked tablet or a
 * dropped WebSocket never stops a line that is already playing.
 *
 * The socket/audio helpers mirror app.js on purpose: the two pages are the
 * only web files in this milestone and share no module of their own.
 */
import { h, render, Component } from './vendor/preact.min.js';
import htm from './vendor/htm.js';

const html = htm.bind(h);
const audio = document.getElementById('audio');

/** Stable per-browser id: the claim survives reloads, so the same device stays the speaker; per role so the Play page in another tab is a different client. */
function clientId() {
  const key = 'bag.client.speaker';
  let id = null;
  try { id = localStorage.getItem(key); } catch { /* private mode */ }
  if (!id) {
    id = Math.random().toString(36).slice(2, 10);
    try { localStorage.setItem(key, id); } catch { /* ignore */ }
  }
  return id;
}

const wsUrl = (id) => `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?client=${id}&role=speaker`;

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

/** Tiny silent WAV: playing it inside the claim tap unlocks the element for later, event-driven playback. */
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

/** Must be called synchronously inside a user gesture. */
function unlockAudio(el) {
  if (!el.paused) return;
  if (!el.src) el.src = silentWav();
  const src = el.src;
  el.play().then(() => { if (el.src === src && src.startsWith('blob:')) el.pause(); }).catch(() => {});
}

class Speaker extends Component {
  state = { claimed: false, wsUp: false, now: null, progress: 0, error: null };
  id = clientId();

  componentDidMount() {
    audio.addEventListener('timeupdate', () => this.setState({ progress: audio.duration ? audio.currentTime / audio.duration : 0 }));
    audio.addEventListener('ended', () => this.ended());
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && this.state.claimed) this.keepAwake();
    });
    this.sock = openSocket(wsUrl(this.id), {
      open: () => { this.setState({ wsUp: true }); if (this.state.claimed) this.sock.send({ type: 'claim' }); },
      close: () => this.setState({ wsUp: false }),
      event: (ev, d) => this.onEvent(ev, d),
    });
  }

  /** The REST claim is the durable one; the socket claim ties this connection to it so presence follows the socket. */
  async claim() {
    unlockAudio(audio);
    try {
      await api('POST', '/api/speaker/claim', { client_id: this.id });
    } catch (e) { return this.setState({ error: e.message }); }
    this.sock.send({ type: 'claim' });
    this.setState({ claimed: true, error: null });
    this.keepAwake();
  }

  /** Screen wake lock keeps a tablet on the couch from sleeping mid-session; it is re-requested on every return to the tab. */
  async keepAwake() {
    if (!navigator.wakeLock) return;
    try { await navigator.wakeLock.request('screen'); } catch { /* denied or tab hidden: nothing to do */ }
  }

  onEvent(ev, d) {
    if (ev === 'status') return this.setState({ now: d.player?.now ?? d.now ?? null });
    if (ev === 'play.start') return this.start(d);
    if (ev === 'play.end') return this.setState({ now: null, progress: 0 });
    if (ev === 'play.stop') {
      audio.pause(); audio.removeAttribute('src'); audio.load();
      return this.setState({ now: null, progress: 0 });
    }
    return undefined;
  }

  start(d) {
    this.setState({ now: d, progress: 0, error: null });
    if (!this.state.claimed) return;
    audio.src = d.url;
    audio.play().catch((e) => this.setState({ error: `cannot play: ${e.message}` }));
  }

  ended() {
    const rid = this.state.now?.render_id;
    if (rid) this.sock.send({ type: 'ended', render_id: rid });
  }

  render(_, s) {
    const error = s.error && html`<div class="banner bad">${s.error}</div>`;
    if (!s.claimed) {
      return html`<main class="speaker">
        <h1>Mr. Bag / speaker</h1>
        <button class="claim" onClick=${() => this.claim()}>Claim speaker</button>
        <p class="muted">This device will play every line. One tap also allows audio.</p>
        ${error}
      </main>`;
    }
    return html`<main class="speaker">
      <h1>Mr. Bag / speaker</h1>
      <div class="now ${s.now ? '' : 'idle'}">${s.now ? s.now.label || s.now.render_id : 'waiting for a line'}</div>
      <div class="sbar"><i style=${`width:${Math.round(s.progress * 100)}%`}></i></div>
      <div class="sstate">
        <span class="dot ${s.wsUp ? 'on' : ''}"><span class="lbl">link</span></span>
        <span>speaker ${this.id}</span>
      </div>
      <button class="stop" onClick=${() => api('POST', '/api/stop').catch((e) => this.setState({ error: e.message }))}>STOP</button>
      ${error}
    </main>`;
  }
}

render(html`<${Speaker} />`, document.getElementById('app'));
