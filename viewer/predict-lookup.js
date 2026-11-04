// Attested-sentence lookup in the browser: a port of src/agents/complete.py (lookup layer only),
// so the Predict page works on static hosting (GitHub Pages) and offline. The Muse Spark
// conjecture layer needs the local server and its API key, and is not ported.
// Loads the corpus from predict-corpus.js (window.UNSCROLL_CORPUS, built by scripts/build_evaluation.py).
(function(){
  var ATTESTED = 0.9, APPROX = 0.8, MIN_WORDS = 3, MAX_FRAGMENT = 600;
  var FOLD = {'ς': 'σ', 'η': 'ε', 'ω': 'ο', 'ϲ': 'σ', 'ϑ': 'θ', 'ϐ': 'β'};
  var WORD = /[\p{L}\p{M}]+[ʼ’']?/gu;

  function fold(word){
    var base = word.toLowerCase().normalize('NFD').replace(/\p{M}/gu, ''), out = '';
    for (var i = 0; i < base.length; i++) { var c = base[i]; if ((c >= 'α' && c <= 'ω') || 'ςϲϑϐ'.indexOf(c) >= 0) out += FOLD[c] || c; }
    return out;
  }
  function words(text){
    var out = [], m; WORD.lastIndex = 0;
    while ((m = WORD.exec(text))) { if (/\d|_/.test(m[0])) continue; var w = fold(m[0]); if (w) out.push([w, m.index, m.index + m[0].length]); }
    return out;
  }

  var DI = [['th', 'θ'], ['ph', 'φ'], ['ch', 'χ'], ['kh', 'χ'], ['ps', 'ψ'], ['rh', 'ρ']];
  var MONO = {}; 'abgdeziklmnxoprstuyfwcq'.split('').forEach(function(c, i){ MONO[c] = 'αβγδεζικλμνξοπρστυυφωκθ'[i]; });
  MONO['ē'] = 'η'; MONO['ê'] = 'η'; MONO['ō'] = 'ω'; MONO['ô'] = 'ω'; MONO['v'] = 'β';
  function transliterate(text){
    if (/\p{Script=Greek}/u.test(text)) return text;
    return text.toLowerCase().split(/([^\p{L}\p{M}]+)/u).map(function(tok){
      if (!tok || !/^\p{L}/u.test(tok)) return tok;
      if (tok[0] === 'h' && tok.length > 1 && 'aeiouyēōêôw'.indexOf(tok[1]) >= 0) tok = tok.slice(1);
      var g = '', i = 0;
      while (i < tok.length) { var two = tok.slice(i, i + 2), hit = null; for (var d = 0; d < DI.length; d++) if (DI[d][0] === two) hit = DI[d][1];
        if (hit) { g += hit; i += 2; continue; } var c = tok[i]; g += c === 'h' ? 'η' : (MONO[c] || c); i++; }
      return g.endsWith('σ') ? g.slice(0, -1) + 'ς' : g;
    }).join('');
  }

  // difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(): Ratcliff/Obershelp on the longest matching block
  function longest(a, al, ah, b, bl, bh){
    var besti = al, bestj = bl, best = 0, j2len = {};
    for (var i = al; i < ah; i++) { var newj = {};
      for (var j = bl; j < bh; j++) { if (a[i] !== b[j]) continue; var k = (j2len[j - 1] || 0) + 1; newj[j] = k; if (k > best) { besti = i - k + 1; bestj = j - k + 1; best = k; } }
      j2len = newj; }
    return [besti, bestj, best];
  }
  function matches(a, al, ah, b, bl, bh){
    var m = longest(a, al, ah, b, bl, bh), k = m[2]; if (!k) return 0;
    return k + (al < m[0] && bl < m[1] ? matches(a, al, m[0], b, bl, m[1]) : 0) + (m[0] + k < ah && m[1] + k < bh ? matches(a, m[0] + k, ah, b, m[1] + k, bh) : 0);
  }
  var simCache = {};
  function wsim(a, b){
    if (a === b) return 1;
    if (Math.abs(a.length - b.length) > Math.floor(Math.max(a.length, b.length) / 2)) return 0;
    var k = a + '\u0000' + b; if (k in simCache) return simCache[k];
    var v = 2 * matches(a, 0, a.length, b, 0, b.length) / (a.length + b.length); return (simCache[k] = v);
  }
  function sim(q, s, first, last){
    var v = wsim(q, s);
    if (q.length >= 2 && q !== s && ((first && s.endsWith(q)) || (last && s.startsWith(q)))) v = Math.max(v, 0.98);
    return v;
  }
  function align(qw, sw){
    var m = qw.length, n = sw.length; if (!m || !n) return [0, 0, 0];
    var INF = 1e9, prev = [], j; for (j = 0; j <= n; j++) prev.push([0, j]);
    for (var i = 1; i <= m; i++) { var cur = [[i, 0]]; for (j = 1; j <= n; j++) cur.push([INF, 0]); var q = qw[i - 1];
      for (j = 1; j <= n; j++) { var sub = 1 - sim(q, sw[j - 1], i === 1, i === m);
        var c1 = [prev[j - 1][0] + (sub < 0.45 ? sub : 1), prev[j - 1][1]], c2 = [prev[j][0] + 1, prev[j][1]], c3 = i > 1 ? [cur[j - 1][0] + 1, cur[j - 1][1]] : [INF, 0];
        var best = c1; if (c2[0] < best[0] || (c2[0] === best[0] && c2[1] < best[1])) best = c2; if (c3[0] < best[0] || (c3[0] === best[0] && c3[1] < best[1])) best = c3; cur[j] = best; }
      prev = cur; }
    var bj = 1; for (j = 2; j <= n; j++) if (prev[j][0] < prev[bj][0]) bj = j;
    return [Math.max(0, 1 - prev[bj][0] / m), prev[bj][1], bj];
  }

  var corpus = null;
  function load(){
    if (corpus) return corpus; var data = window.UNSCROLL_CORPUS; if (!data) return null;
    var ws = data.sentences.map(function(s){ return words(s.text); }), index = {};
    ws.forEach(function(w, i){ var seen = {}; w.forEach(function(t){ if (!seen[t[0]]) { seen[t[0]] = 1; (index[t[0]] = index[t[0]] || []).push(i); } }); });
    return (corpus = {source: data.source, sents: data.sentences, words: ws, index: index, vocab: Object.keys(index)});
  }
  function candidates(c, qw, limit){
    var counts = {}, uniq = {}; qw.forEach(function(w){ uniq[w] = 1; });
    Object.keys(uniq).forEach(function(w){ var ids = c.index[w];
      if (!ids && w !== qw[0] && w !== qw[qw.length - 1]) { var near = w.length > 3 ? c.vocab.filter(function(v){ return v.length > 3 && wsim(w, v) >= 0.8; }).slice(0, 5) : []; ids = []; near.forEach(function(v){ ids = ids.concat(c.index[v]); }); }
      ids = ids || []; var weight = 1 / Math.sqrt(1 + ids.length), seen = {};
      ids.forEach(function(i){ if (seen[i]) return; seen[i] = 1; counts[i] = (counts[i] || 0) + weight; }); });
    if (qw.length <= 2 || !Object.keys(counts).length) {
      c.words.forEach(function(ws, k){ var lw = ws.map(function(t){ return t[0]; });
        if (lw.some(function(w){ return w.endsWith(qw[0]); }) && (qw.length === 1 || lw.some(function(w){ return w.startsWith(qw[qw.length - 1]); }))) counts[k] = (counts[k] || 0) + 0.5; });
    }
    return Object.keys(counts).sort(function(a, b){ return counts[b] - counts[a]; }).slice(0, limit || 300).map(Number);
  }
  function lookup(fragment, top){
    var c = load(); if (!c) throw new Error('Corpus not loaded');
    var qw = words(fragment).map(function(t){ return t[0]; }); if (!qw.length) return [];
    var hits = [];
    candidates(c, qw).forEach(function(i){ var sw = c.words[i], r = align(qw, sw.map(function(t){ return t[0]; }));
      if (r[0] >= APPROX) { var s = c.sents[i], lo = sw[r[1]][1], hi = sw[r[2] - 1][2];
        hits.push({id: s.id, ref: s.ref, text: s.text, score: Math.round(r[0] * 1000) / 1000, before: s.text.slice(0, lo), matched: s.text.slice(lo, hi), after: s.text.slice(hi)}); } });
    hits.sort(function(a, b){ return b.score - a.score; });
    return hits.slice(0, top || 5);
  }
  function complete(fragment){
    if (typeof fragment !== 'string' || !fragment.trim() || fragment.length > MAX_FRAGMENT) throw new Error('Enter 1–' + MAX_FRAGMENT + ' characters of Greek (or Latin transliteration)');
    var greek = transliterate(fragment.trim()).normalize('NFC'), qw = words(greek);
    if (!qw.length) throw new Error('No Greek words found');
    var c = load(), hits = lookup(greek), out = {query: fragment.trim(), greek: greek, words: qw.length, corpus: c.source};
    if (hits.length) { var best = hits[0].score, tied = hits.filter(function(h){ return h.score >= best - 0.02; }), texts = {}; tied.forEach(function(h){ texts[h.text] = 1; });
      out.status = qw.length < MIN_WORDS ? 'too_short' : best >= ATTESTED ? 'attested' : 'attested_approx'; out.matches = hits; out.ambiguous = Object.keys(texts).length > 1; return out; }
    out.status = 'not_found'; out.matches = []; out.note = 'No attested sentence contains this fragment. Muse Spark conjecture needs the local server (Launch Unscroll.cmd).';
    return out;
  }
  window.UNSCROLL_LOOKUP = {complete: complete, lookup: lookup, words: words, transliterate: transliterate, wsim: wsim, align: align, ready: function(){ return !!load(); }};
})();
