// Hands-free review: push-to-talk -> local server -> Muse Voice Transcribe -> workbench action.
// Records 16 kHz mono 16-bit WAV in the browser; the API key stays on the local server.
(function () {
  const $ = id => document.getElementById(id);

  // Transcript -> {action, value, label}. Pure, so it can be tested without a microphone.
  function parseCommand(text) {
    const t = (text || '').toLowerCase().replace(/[.,!?]/g, ' ').trim();
    const has = (...w) => w.some(x => t.includes(x));
    const n = t.match(/\b(?:slice|go to|number)\s+(\d{1,3})\b/);
    if (n) return { action: 'slice', value: +n[1], label: 'Slice ' + n[1] };
    if (has('worst', 'largest error', 'biggest error', 'மோசமான')) return { action: 'worst', label: 'Jump to the largest error' };
    if (has('pause', 'stop', 'நிறுத்து')) return { action: 'pause', label: 'Pause' };
    if (has('play', 'start', 'தொடங்கு')) return { action: 'play', label: 'Play' };
    const show = has('show', 'turn on') ? true : has('hide', 'turn off') ? false : null;
    if (show !== null && has('previous trace', 'old trace', 'old')) return { action: 'toggle', value: ['old', show], label: (show ? 'Show' : 'Hide') + ' previous trace' };
    if (show !== null && has('mesh', 'reference')) return { action: 'toggle', value: ['mesh', show], label: (show ? 'Show' : 'Hide') + ' released mesh' };
    if (show !== null && has('trace')) return { action: 'toggle', value: ['trace', show], label: (show ? 'Show' : 'Hide') + ' trace' };
    if (has('mark', 'flag', 'குறி')) return { action: 'mark', label: 'Toggle review marking' };
    if (has('next', 'forward', 'அடுத்து', 'அடுத்த')) return { action: 'next', label: 'Next slice' };
    if (has('back', 'previous', 'முந்தைய', 'பின்')) return { action: 'prev', label: 'Previous slice' };
    return { action: 'none', label: 'No command recognised' };
  }

  function key(k) { document.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true })); }
  function apply(cmd) {
    const playing = $('play') && $('play').textContent.trim() === 'Ⅱ';
    switch (cmd.action) {
      case 'next': key('ArrowRight'); break;
      case 'prev': key('ArrowLeft'); break;
      case 'worst': $('worst') && $('worst').click(); break;
      case 'play': if (!playing) $('play').click(); break;
      case 'pause': if (playing) $('play').click(); break;
      case 'slice': { const s = $('slice'); if (s) { s.value = Math.max(0, cmd.value - 1); s.dispatchEvent(new Event('input')); } break; }
      case 'toggle': { const [id, on] = cmd.value, c = $(id); if (c) { c.checked = on; c.dispatchEvent(new Event('change')); } break; }
      case 'mark': { const c = $('mark'); if (c) { c.checked = !c.checked; c.dispatchEvent(new Event('change')); } break; }
    }
  }

  function wav16k(chunks, rate) {
    let len = chunks.reduce((a, c) => a + c.length, 0), pcm = new Float32Array(len), o = 0;
    for (const c of chunks) { pcm.set(c, o); o += c.length; }
    if (rate !== 16000) {                               // linear resample if the context ignored 16 kHz
      const n = Math.floor(pcm.length * 16000 / rate), r = new Float32Array(n);
      for (let i = 0; i < n; i++) { const x = i * rate / 16000, j = Math.floor(x), f = x - j; r[i] = (pcm[j] || 0) * (1 - f) + (pcm[j + 1] || 0) * f; }
      pcm = r;
    }
    const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf), w = (p, s) => [...s].forEach((ch, i) => v.setUint8(p + i, ch.charCodeAt(0)));
    w(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); w(8, 'WAVE'); w(12, 'fmt '); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true); v.setUint32(28, 32000, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
    return new Blob([buf], { type: 'audio/wav' });
  }

  let rec = null;
  async function start(btn, out) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext({ sampleRate: 16000 }), src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1), chunks = [];
      proc.onaudioprocess = e => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(proc); proc.connect(ctx.destination);
      rec = { stream, ctx, proc, chunks };
      btn.textContent = '● Listening… click to send'; btn.setAttribute('aria-pressed', 'true');
      out.textContent = 'Say: next · back · slice 20 · worst · play · pause · show mesh · mark';
    } catch (e) { out.textContent = 'Microphone unavailable: use arrows and space.'; }
  }
  async function stop(btn, out) {
    const { stream, ctx, proc, chunks } = rec; rec = null;
    proc.disconnect(); stream.getTracks().forEach(t => t.stop()); await ctx.close();
    btn.textContent = '🎙 Voice'; btn.setAttribute('aria-pressed', 'false'); out.textContent = 'Transcribing with Muse Voice…';
    try {
      const r = await fetch('/api/voice/transcribe', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav16k(chunks, ctx.sampleRate) });
      const j = await r.json();
      if (!r.ok) { out.textContent = j.error || 'Voice service unavailable.'; return; }
      const cmd = parseCommand(j.transcript);
      out.textContent = '“' + (j.transcript || '…') + '” → ' + cmd.label;
      apply(cmd);
    } catch (e) { out.textContent = 'Voice service unavailable. Use the keyboard: arrows, space.'; }
  }

  function mount() {
    const header = document.querySelector('header');
    if (!header || $('voiceBtn')) return;
    const btn = document.createElement('button'); btn.id = 'voiceBtn'; btn.type = 'button';
    btn.textContent = '🎙 Voice'; btn.setAttribute('aria-pressed', 'false');
    btn.title = 'Push to talk: Muse Voice Transcribe';
    const out = document.createElement('span'); out.id = 'voiceOut'; out.setAttribute('role', 'status');
    out.style.cssText = 'font-size:12px;color:var(--muted);max-width:360px';
    btn.onclick = () => (rec ? stop(btn, out) : start(btn, out));
    header.insertBefore(out, header.querySelector('.local'));
    header.insertBefore(btn, out);
  }
  window.UnscrollVoice = { parseCommand, apply };
  mount();
})();
