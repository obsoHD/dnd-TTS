# Vendored front-end libraries

Pinned, unmodified builds downloaded from jsdelivr so the table works offline.

| file | package | version | source | sha256 (first 16) |
|---|---|---|---|---|
| `preact.min.js` | preact | 10.29.8 | https://cdn.jsdelivr.net/npm/preact@10.29.8/dist/preact.module.js | `c30e721ebfdc6e2a` |
| `htm.js` | htm | 3.1.1 | https://cdn.jsdelivr.net/npm/htm@3.1.1/dist/htm.module.js | `ab33dd3f38059b9b` |

Both are ES modules (`preact.min.js` is the minified `preact.module.js` build), imported directly by
`web/app.js` and `web/speaker.js`; there is no build step.

`preact/hooks` is deliberately not vendored: the pages keep their state in one `Component` subclass and
every child is a pure function of props, so the core build is enough.

## Fallback if these files are missing

Replace the two imports at the top of `app.js` / `speaker.js` with the CDN URLs above; nothing else changes.
The pages then require internet access on the tablet, which is why the offline copies are preferred.
