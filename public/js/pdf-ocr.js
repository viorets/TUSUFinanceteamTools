// OCR for PDF pages (OpenThai 2.1 OCR Text by iApp, v2.1.4; PaddleOCR models run in the browser with onnxruntime-web):
// finds the text lines on a rendered page (PP-OCRv6 tiny detection) and reads each line (OpenThai recognition: Thai,
// Thai digits, English, digits). Returns pdf.js-like text items in PDF units, so js/pdf-text.js builds paragraphs,
// lists and tables from them as usual. 2.1 keeps 2.0's det.onnx and rec.onnx; what it adds is ported here from its
// Python runtime (runtime/line_engine.py): rec2.onnx for loopless fonts, floating tone marks, decimal commas.
// Engine (public/vendor/ort/) and models (public/models/ocr/openthai2/) are served from this site and load on first
// use only. A new model goes in a new folder: models are cached for 30 days (firebase.json).

import { fixThai, loadWords } from './thai-check.js';

const ORT = new URL('../vendor/ort/', import.meta.url).href; // onnxruntime-web 1.20.1, served from this site
const MODELS = new URL('../models/ocr/openthai2/', import.meta.url).href;
const MAX_SIDE = 1600; // detection input, longest side (px)
const READ_SIDE = 2800; // the page as rendered for reading lines (sharper small marks)
const REC_H = 48; // recognition line height
// detection limits: text pixels above THRESH; keep regions whose mean is over BOX_THRESH; grow them by UNCLIP.
// OpenThai's own det.yml values (0.2 / 0.4 / 1.4) scored the same on 23 test pages (14 Nov 2026), so these stay.
const THRESH = 0.3; const BOX_THRESH = 0.6; const UNCLIP = 1.6;

let ready = null;
// Loads onnxruntime-web, both models and the character list (once). onProgress(text) reports what is loading.
export function loadOcr(onProgress = () => {}) {
  return ready ||= (async () => {
    if (!window.ort) {
      onProgress('Loading the OCR engine…');
      await new Promise((res, rej) => { const s = document.createElement('script'); s.src = `${ORT}ort.min.js`; s.crossOrigin = 'anonymous'; s.onload = res; s.onerror = () => rej(new Error('Could not load the OCR engine. Check the internet connection and try again.')); document.head.append(s); });
    }
    const { ort } = window; ort.env.wasm.wasmPaths = ORT;
    onProgress('Loading the Thai OCR models (about 4 MB, first time only)…');
    const [det, rec, dict] = await Promise.all([
      ort.InferenceSession.create(`${MODELS}det.onnx`),
      ort.InferenceSession.create(`${MODELS}rec.onnx`),
      fetch(`${MODELS}rec_dict.txt`).then((r) => { if (!r.ok) throw new Error('OCR character list missing'); return r.text(); }),
    ]);
    // class 0 is the CTC blank; then one class per dictionary line; the last class is a space
    const chars = ['', ...dict.replace(/\r/g, '').split('\n').filter((c, i, a) => c !== '' || i < a.length - 1), ' '];
    return { ort, det, rec, chars };
  })().catch((e) => { ready = null; throw e; });
}

// One pdf.js page -> text items { str, transform: [s, 0, 0, s, x, y], width, height, fontName } in PDF units.
export async function ocrPage(page, onProgress = () => {}) {
  const m = await loadOcr(onProgress);
  const base = page.getViewport({ scale: 1 });
  const scale = Math.min(READ_SIDE / Math.max(base.width, base.height), 4);
  const vp = page.getViewport({ scale });
  const cv = document.createElement('canvas'); cv.width = Math.round(vp.width); cv.height = Math.round(vp.height);
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
  await page.render({ canvasContext: ctx, viewport: vp }).promise;
  const all = await detect(m, cv);
  // Detection sometimes cuts a few marks off a line into a small box of their own, between two lines; read alone
  // they give stray "lines" ("ะ*", "9") that split paragraphs. A box under ~half the usual height that lies in
  // the band of a full-height box it overlaps is such a piece.
  const hOf = (b) => b.y1 - b.y0; const med = mid(all.map(hOf));
  const full = all.filter((b) => hOf(b) >= med * 0.55);
  const boxes = all.filter((b) => hOf(b) >= med * 0.55 || !full.some((o) => o.x0 < b.x1 && b.x0 < o.x1 && b.y0 < o.y1 + med * 0.5 && b.y1 > o.y0 - med * 0.5));
  const texts = new Map();
  for (const b of boxes) texts.set(b, await readLine(m, cv, b));
  await attachMarks(m, cv, all.filter((b) => !boxes.includes(b)), boxes, texts);
  const read = [];
  for (const b of boxes) {
    const str = texts.get(b).trim().replace(DECIMAL_COMMA, '$1$2.$3');
    if (!str) continue;
    const [x0, y0] = vp.convertToPdfPoint(b.x0, b.y1); const [x1, y1] = vp.convertToPdfPoint(b.x1, b.y0);
    const h = Math.abs(y1 - y0);
    read.push({ str, x: Math.min(x0, x1), w: Math.abs(x1 - x0), mid: (y0 + y1) / 2, size: h * 0.72 }); // the box covers marks above and below
  }
  // A box's height depends on which marks its words happen to have, so sizes and baselines jump from line to line
  // and body lines read as headings or new paragraphs. Boxes on one line share its size and middle; sizes within
  // half of the page's usual size are that size (a box's height says little about its type size: a title without
  // marks measures like body text). Baseline: the box middle less 0.39 × size (a full Thai line box).
  const body = mid(read.map((r) => r.size));
  const lines = [];
  [...read].sort((a, b) => b.mid - a.mid).forEach((r) => {
    const ln = lines.at(-1);
    if (ln && Math.abs(ln[0].mid - r.mid) < body * 0.5) ln.push(r); else lines.push([r]);
  });
  for (const ln of lines) {
    let size = mid(ln.map((r) => r.size)); if (Math.abs(size / body - 1) < 0.5) size = body;
    const y = mid(ln.map((r) => r.mid)) - size * 0.39;
    ln.forEach((r) => { r.size = size; r.y = y; });
  }
  // Each line's Thai is checked against the word list: a word missing one mark is put right; words close to real ones
  // but not changed are returned in items.check ([word, suggestion]) for the user to look at.
  let dict = null; try { dict = await loadWords(); } catch { /* no word list: the lines stay as read */ }
  const check = []; let fixed = 0;
  const items = read.map((r) => { const f = fixThai(r.str, dict); fixed += f.fixed.length; check.push(...f.unsure); return { str: f.text, transform: [r.size, 0, 0, r.size, r.x, r.y], width: r.w, height: r.size, fontName: 'ocr' }; });
  return Object.assign(items, { check, fixed });
}
// 2.1.4: an amount whose last comma has exactly two digits after it ("88,226,21", "280,00") had a decimal point
const DECIMAL_COMMA = /(^|[^0-9๐-๙.,])([0-9๐-๙]{1,3}(?:,[0-9๐-๙]{3})*),([0-9๐-๙]{2})(?![0-9๐-๙.,])/g;
const mid = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0; };

// ---------- detection (DB): probability map -> line boxes ----------
async function detect({ ort, det }, cv) {
  const k = Math.min(1, MAX_SIDE / Math.max(cv.width, cv.height)); // detection works on a smaller copy
  const W = Math.max(32, Math.round((cv.width * k) / 32) * 32); const H = Math.max(32, Math.round((cv.height * k) / 32) * 32);
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(cv, 0, 0, W, H);
  const px = g.getImageData(0, 0, W, H).data; const n = W * H; const data = new Float32Array(3 * n);
  const mean = [0.485, 0.456, 0.406]; const std = [0.229, 0.224, 0.225];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) data[k * n + i] = (px[i * 4 + k] / 255 - mean[k]) / std[k];
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, H, W]) });
  const prob = out[det.outputNames[0]].data;
  // connected regions above THRESH; keep those whose mean probability is over BOX_THRESH
  const seen = new Uint8Array(n); const boxes = []; const stack = [];
  for (let i = 0; i < n; i++) {
    if (seen[i] || prob[i] < THRESH) continue;
    let x0 = W, y0 = H, x1 = 0, y1 = 0, sum = 0, cnt = 0; stack.push(i); seen[i] = 1;
    while (stack.length) {
      const j = stack.pop(); const x = j % W; const y = (j - x) / W;
      sum += prob[j]; cnt++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (const k of [j - 1, j + 1, j - W, j + W]) {
        if (k < 0 || k >= n || seen[k] || prob[k] < THRESH) continue;
        if ((k === j - 1 && x === 0) || (k === j + 1 && x === W - 1)) continue;
        seen[k] = 1; stack.push(k);
      }
    }
    const w = x1 - x0 + 1; const h = y1 - y0 + 1;
    if (cnt < 10 || Math.min(w, h) < 3 || sum / cnt < BOX_THRESH) continue;
    // unclip: DB shrinks text regions; grow by area * UNCLIP / perimeter on every side
    const d = (w * h * UNCLIP) / (2 * (w + h));
    const sx = cv.width / W; const sy = cv.height / H;
    // pad: extra width read at each end, so a narrow first letter (เ, ไ) is not cut off; the box itself (used for
    // layout) stays tight
    boxes.push({ x0: Math.max(0, (x0 - d) * sx), y0: Math.max(0, (y0 - d) * sy), x1: Math.min(cv.width, (x1 + 1 + d) * sx), y1: Math.min(cv.height, (y1 + 1 + d) * sy), pad: (h + 2 * d) * 0.3 * sx });
  }
  return boxes.sort((a, b) => a.y0 - b.y0 || a.x0 - b.x0);
}

// ---------- floating marks (2.1): marks the detector boxed apart from their line ----------
// Old letterpress books (and some display fonts) float tone marks and upper vowels so far above the letters that the
// line's box leaves them out; their small boxes are dropped above, so the marks were lost (2.0 lost 128 of 159 on a
// 1935 page). Each such box within 0.35 line heights above (or below) a line it overlaps goes to the nearest line,
// which is read again grown to take it in. If a quarter of the lines (at least 3) had marks, the typeface floats
// them, so every other line grows by the median reach too, short of its neighbour. A re-read is kept only if it
// changes no digit and, unless the typeface floats its marks, nothing but marks. Only the text changes: the line's
// box (layout) stays. Limits as in the runtime's _attach_marks (max_gap 0.35, max_ext 0.9, max_height 0.9).
async function attachMarks(m, cv, marks, boxes, texts) {
  if (!marks.length) return;
  const hmed = mid(boxes.map((b) => b.y1 - b.y0));
  const lines = boxes.filter((b) => b.y1 - b.y0 >= 0.5 * hmed);
  const reach = new Map(lines.map((a) => [a, [0, 0]])); // [above, below], px
  for (const j of marks) {
    let best = null;
    for (const a of lines) {
      const ha = a.y1 - a.y0;
      if (j.y1 - j.y0 > 0.9 * ha || Math.min(j.x1, a.x1) - Math.max(j.x0, a.x0) < 0.5 * (j.x1 - j.x0)) continue;
      const t0 = j.y0 - a.y0; const t1 = j.y1 - a.y0;
      let gap; let side; let ext;
      if (t1 <= 0.5 * ha) [gap, side, ext] = [Math.max(0, -t1), 0, -t0];
      else if (t0 >= 0.5 * ha) [gap, side, ext] = [Math.max(0, t0 - ha), 1, t1 - ha];
      else continue; // across the line's middle: part of the line
      if (gap > 0.35 * ha || ext > 0.9 * ha) continue;
      if (!best || gap < best.gap || (gap === best.gap && side < best.side)) best = { gap, side, ext, a }; // a tie: the line below the mark
    }
    if (best) reach.get(best.a)[best.side] = Math.max(reach.get(best.a)[best.side], best.ext);
  }
  let floating = false;
  for (const side of [0, 1]) {
    const lifted = lines.filter((a) => reach.get(a)[side] > 0);
    if (!lifted.length || lifted.length < Math.max(3, 0.25 * lines.length)) continue;
    floating = true;
    const typ = mid(lifted.map((a) => reach.get(a)[side] / (a.y1 - a.y0)));
    for (const a of lines) {
      if (reach.get(a)[side] > 0) continue;
      const half = a.y0 + 0.5 * (a.y1 - a.y0);
      let room = typ * (a.y1 - a.y0);
      for (const o of lines) { // stop short of the neighbouring line
        if (o === a || Math.min(o.x1, a.x1) - Math.max(o.x0, a.x0) <= 0) continue;
        if (side === 0 && o.y1 <= half) room = Math.min(room, Math.max(0, (a.y0 - o.y1) * 0.9));
        if (side === 1 && o.y0 >= half) room = Math.min(room, Math.max(0, (o.y0 - a.y1) * 0.9));
      }
      reach.get(a)[side] = room;
    }
  }
  const digits = (t) => t.replace(/[^0-9๐-๙]/g, '');
  const base = (t) => t.replace(/ำ/g, 'า').replace(/[ัิ-ฺ็-๎\s]/g, ''); // without marks
  for (const a of lines) {
    const [up, down] = reach.get(a); if (up <= 0 && down <= 0) continue;
    const str = await readLine(m, cv, { ...a, y0: Math.max(0, a.y0 - up), y1: Math.min(cv.height, a.y1 + down) });
    const old = texts.get(a);
    if (str.trim() && (floating ? digits(str) === digits(old) : base(str) === base(old))) texts.set(a, str);
  }
}

// ---------- recognition (CTC): one line box -> text ----------
// A line with Latin letters inside a Thai word (a loopless font such as Kanit, Prompt or Sukhumvit Set read as
// look-alikes: "lละพลNSะทU") is read again by rec2.onnx (2.1.2, the same network trained with loopless fonts too), and
// rec2's reading is kept if it has fewer of them. rec2 (2.4 MB) loads the first time a line needs it.
let rec2 = null;
async function readLine(m, cv, b) {
  const x = lineTensor(m.ort, cv, b); if (!x) return '';
  const s = await ctc(m, m.rec, x); const n = mixed(s); if (!n) return s;
  try {
    rec2 ||= m.ort.InferenceSession.create(`${MODELS}rec2.onnx`).catch((e) => { rec2 = null; throw e; });
    const alt = await ctc(m, await rec2, x);
    return mixed(alt) < n ? alt : s;
  } catch { return s; } // rec2 could not load (offline): the first reading, as in 2.0
}
// Latin letters inside words that also hold Thai letters; a Thai word next to an English one does not count
const mixed = (t) => t.split(/\s+/).reduce((n, w) => n + (/[ก-๎]/.test(w) ? (w.match(/[A-Za-z]/g) || []).length : 0), 0);

function lineTensor(ort, cv, b) {
  const x0 = Math.max(0, b.x0 - (b.pad || 0)); const x1 = Math.min(cv.width, b.x1 + (b.pad || 0));
  const bw = x1 - x0; const bh = b.y1 - b.y0; if (bw < 2 || bh < 2) return null;
  const W = Math.min(Math.max(Math.round((bw / bh) * REC_H), 16), 3200);
  const c = document.createElement('canvas'); c.width = W; c.height = REC_H;
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(cv, x0, b.y0, bw, bh, 0, 0, W, REC_H);
  const px = g.getImageData(0, 0, W, REC_H).data; const n = W * REC_H; const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) data[k * n + i] = (px[i * 4 + (2 - k)] / 255 - 0.5) / 0.5; // BGR, as Paddle trains
  return new ort.Tensor('float32', data, [1, 3, REC_H, W]);
}

async function ctc({ chars }, rec, x) {
  const out = await rec.run({ [rec.inputNames[0]]: x });
  const t = out[rec.outputNames[0]]; const [, T, C] = t.dims; const p = t.data;
  let s = ''; let last = 0;
  for (let i = 0; i < T; i++) {
    let best = 0; let bv = -Infinity;
    for (let k = 0; k < C; k++) { const v = p[i * C + k]; if (v > bv) { bv = v; best = k; } }
    if (best !== last && best !== 0) s += chars[best] ?? '';
    last = best;
  }
  return s;
}
