'use strict';
// Full page from CT: Unscroll's run of the Grand Prize ink model over a whole released segment,
// with Muse Spark's unscored reading of its readable lines (scripts/read_fullpage.py).
(() => {
  const F = window.UNSCROLL_FULLPAGE, $ = id => document.getElementById(id);
  if (!F || !$('fullPage')) return;
  const ns = 'http://www.w3.org/2000/svg', [W, H] = F.size, ink = F.ink;
  $('fpIntro').textContent = 'Unscroll streamed all 26 surface layers of released segment ' + F.segment +
    ' (' + ink.shape[0] + ' × ' + ink.shape[1] + ' px) on a Kaggle T4 and ran the 2023 Grand Prize ink model over every tile in ' +
    ink.minutes + ' minutes. White ink became dark here; the page is rotated, mirrored (the segment faces the back of the sheet) and levelled by ' +
    F.deskew_deg + '° so the Greek reads left to right. Not built by Unscroll: the ink model (2023 Vesuvius Challenge Grand Prize, MIT) and this segment (traced by the Vesuvius Challenge community). Built by Unscroll: running it on Kaggle, turning the page upright, the line boxes and the Spark reading below.';
  $('fpImg').src = F.image;
  const svg = $('fpBoxes'); svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  F.lines.forEach((l, i) => {
    const [x0, y0, x1, y1] = l.bbox, r = document.createElementNS(ns, 'rect');
    for (const [k, v] of [['x', x0], ['y', y0], ['width', x1 - x0], ['height', y1 - y0]]) r.setAttribute(k, v);
    svg.append(r);
    const t = document.createElementNS(ns, 'text'); t.setAttribute('x', x1 - 30); t.setAttribute('y', y0 + 70); t.textContent = i + 1; svg.append(t);
    const li = document.createElement('li'), g = document.createElement('span'); g.className = 'greek';
    g.textContent = l.text || '[left unread]'; li.append(g); $('fpLines').append(li);
  });
  const cer = F.published_column_cer != null ? Math.round(F.published_column_cer * 100) + '%' : 'high';
  $('fpCaption').textContent = 'Lines: ' + F.model + ' reading only these four crops, ' + F.seconds + ' s. Unscored: no published transcription is aligned to this sheet. ' +
    'On the published columns above the same model had ' + cer + ' character error, so treat these as rough letter guesses for a papyrologist to check, not a reading.';
  $('fullPage').hidden = false;
})();
