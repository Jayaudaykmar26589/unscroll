'use strict';
(() => {
  const b=window.UNSCROLL_READING,R=window.ReadingState,$=id=>document.getElementById(id);
  if(!b||!R)return;
  const key='unscroll-reading-'+b.bundle_id;let state,selected=b.lines[0].id,busy=false;
  try{state=R.restore(b,JSON.parse(localStorage.getItem(key)||'null'));}catch{state=R.create(b);}
  const translationDirty={en:false,ta:false};
  const ns='http://www.w3.org/2000/svg';
  $('passageTitle').textContent=b.title;$('disclosure').textContent=b.source.relationship_to_experiment;
  $('sourceLink').href=b.source.url;$('imageLink').href=b.source.image_url;$('imageCredit').textContent=b.source.attribution;
  // passage picker: every published column, plus the original five-line excerpt
  for(const p of window.UNSCROLL_READINGS||[b]){const a=document.createElement('a');a.href='read.html?p='+encodeURIComponent(p.id);a.textContent=p.id===(window.UNSCROLL_READINGS||[b])[0].id?'Col. −8 · ll. 10–14 (excerpt)':p.title.replace('PHerc. Paris. 4 · ','').replace('column','Col.').replace('lines','ll.');
    if(p.id===b.id){a.className='active';a.setAttribute('aria-current','page');}$('passagePicker').append(a);}
  if(!b.recognition_supported){const sec=document.querySelector('section[aria-label="Image recognition"]');sec.hidden=true;$('recognitionOff').hidden=false;
    const cr=b.column_reading;if(cr)$('recognitionOff').textContent+=` Measured: ${cr.model}, reading only the line crops of this column, had ${(cr.normalized_cer*100).toFixed(0)}% character error against the published transcription (${cr.exact_lines}/${cr.lines} lines exact). Reading is not yet automatic.`;}
  if(b.lines.length>6)$('wholeColumn').checked=true;
  $('columnName').textContent='Column '+(b.source.location.match(/column (−?\d+)/)||['','−8'])[1];
  $('editorialNote').textContent=b.editorial_note;$('bundleHash').textContent='Source bundle SHA-256: '+b.bundle_id;
  function persist(){try{localStorage.setItem(key,JSON.stringify(state));$('storageStatus').textContent='Saved in this browser. Export to keep a portable copy.';}catch{$('storageStatus').textContent='Browser storage unavailable. Export this reading before closing.';}}
  function dirtyGreek(){return $('greekEditor').value!==state.lines[selected];}
  function pending(){return dirtyGreek()||translationDirty.en||translationDirty.ta;}
  function pick(id){if(pending()){$('editNotice').textContent='Save your pending edits before changing lines.';return;}selected=id;render();}
  function drawImage(){const svg=document.createElementNS(ns,'svg');const W=b.source.image_width,H=b.source.image_height,top=Math.max(Math.min(...b.lines.map(l=>l.bbox[1]))-15,0),bot=Math.min(Math.max(...b.lines.map(l=>l.bbox[3]))+30,H);svg.setAttribute('viewBox',$('wholeColumn').checked?`0 0 ${W} ${H}`:`0 ${top} ${W} ${bot-top}`);svg.setAttribute('role','group');svg.setAttribute('aria-label','Published column image with selectable line regions');
    const im=document.createElementNS(ns,'image');im.setAttribute('href',b.source.image_path);im.setAttribute('width',b.source.image_width);im.setAttribute('height',b.source.image_height);svg.append(im);
    for(const line of b.lines){const [x0,y0,x1,y1]=line.bbox,rect=document.createElementNS(ns,'rect');rect.setAttribute('x',x0);rect.setAttribute('y',y0);rect.setAttribute('width',x1-x0);rect.setAttribute('height',y1-y0);rect.setAttribute('class',line.id===selected?'active':'');rect.setAttribute('role','button');rect.setAttribute('tabindex','0');rect.setAttribute('aria-label','Select image line '+line.label);rect.onclick=()=>pick(line.id);rect.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick(line.id);}};svg.append(rect);
      const text=document.createElementNS(ns,'text');text.setAttribute('x',x1-40);text.setAttribute('y',y0+25);text.textContent=line.label.split(' ')[0];svg.append(text);}
    $('imageStage').replaceChildren(svg);}
  function cards(){$('lineCards').replaceChildren();for(const line of b.lines){const btn=document.createElement('button');btn.className='lineCard'+(line.id===selected?' active':'');btn.setAttribute('aria-label','Select transcription line '+line.label);btn.setAttribute('aria-pressed',String(line.id===selected));
    for(const [cls,text] of [['number',line.label],['greek',state.lines[line.id]||'[blank correction]'],['edited',state.lines[line.id]!==line.text?'edited':'']]){const span=document.createElement('span');span.className=cls;span.textContent=text;if(cls==='greek')span.lang='grc';btn.append(span);}btn.onclick=()=>pick(line.id);$('lineCards').append(btn);}}
  function statuses(){const dirty=pending();
    $('recognizeLocal').disabled=busy||location.protocol==='file:';$('recordedLocal').disabled=busy||!window.UNSCROLL_LOCAL_OCR;
    $('recognizeLive').disabled=busy||location.protocol==='file:';$('recognizeRecorded').disabled=busy||!window.UNSCROLL_RECOGNITION;
    $('acceptPrediction').disabled=dirty||busy||!state.recognition?.lines.some(l=>l.id===selected);
    $('predictionText').textContent=state.recognition?.lines.find(l=>l.id===selected)?.text||'No readable prediction loaded for this region.';$('exportReading').disabled=dirty;$('translateLive').disabled=dirty||busy||location.protocol==='file:';
    $('recordedSpark').disabled=dirty||busy||!b.recorded_model||R.norm(R.source(b,state))!==R.norm(b.source_text);
    for(const k of ['en','ta']){const valid=R.current(b,state,k)&&!dirtyGreek();$(k+'State').textContent=valid?'Current draft':'Stale — Greek changed';$(k+'State').className='badge'+(valid?'':' stale');$(k+'Provenance').textContent=state.translations[k].kind.replaceAll('_',' ')+' · '+state.translations[k].notes;}
    if(location.protocol==='file:')$('translationNotice').textContent='Offline file mode. Saved drafts and human edits work; launch the local server to request Spark translations.';
  }
  function render(){const line=b.lines.find(l=>l.id===selected);$('activeLineLabel').textContent='line '+line.label;$('editorLabel').textContent='Your transcription · line '+line.label;$('greekEditor').value=state.lines[selected];$('publishedLine').textContent='Published: '+line.text;$('revision').textContent='Revision '+state.revision;
    cards();drawImage();for(const k of ['en','ta'])if(!translationDirty[k])$(k+'Text').value=state.translations[k].text;
    $('history').replaceChildren();if(!state.history.length){const li=document.createElement('li');li.textContent='No corrections yet. Published text and saved translation drafts are intact.';$('history').append(li);}
    for(const h of state.history.slice().reverse()){const li=document.createElement('li');li.textContent=h.type==='transcription'?`r${h.revision} · ${h.line_id}: ${h.before} → ${h.after||'[blank]'}`:h.type==='recognition_accept'?`r${h.revision} · ${h.line_id}: image prediction applied (${h.model}); ${h.before} → ${h.after||'[unreadable]'}`:h.type==='translation'?`r${h.revision} · ${h.target} human translation edit`:`r${h.revision} · ${h.model} translation draft`;$('history').append(li);}statuses();}
  $('wholeColumn').onchange=drawImage;
  $('greekEditor').oninput=()=>{$('editNotice').textContent=dirtyGreek()?'Unsaved correction. Save to update the reading and translation status.':'';statuses();};
  $('saveLine').onclick=()=>{if(translationDirty.en||translationDirty.ta){$('editNotice').textContent='Save translation edits before changing their Greek source.';return;}try{const changed=R.edit(b,state,selected,$('greekEditor').value);persist();render();$('editNotice').textContent=changed?'Correction saved. Previous translations are stale until retranslated or reviewed against this reading.':'No change to save.';}catch(e){$('editNotice').textContent=e.message;}};
  $('restoreLine').onclick=()=>{if(translationDirty.en||translationDirty.ta){$('editNotice').textContent='Save translation edits first.';return;}R.edit(b,state,selected,b.lines.find(l=>l.id===selected).text);persist();render();$('editNotice').textContent='Published line restored; the audit history is retained.';};
  for(const k of ['en','ta']){$(k+'Text').oninput=()=>{translationDirty[k]=$(k+'Text').value!==state.translations[k].text;statuses();};$('save'+(k==='en'?'En':'Ta')).onclick=()=>{if(dirtyGreek()){$('translationNotice').textContent='Save the Greek correction before attaching a translation.';return;}try{R.humanTranslation(b,state,k,$(k+'Text').value);translationDirty[k]=false;persist();render();$('translationNotice').textContent='Human translation draft saved against the current Greek reading.';}catch(e){$('translationNotice').textContent=e.message;}};}
  async function hash(text){const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(R.norm(text)));return Array.from(new Uint8Array(bytes),v=>v.toString(16).padStart(2,'0')).join('');}
  $('recordedSpark').onclick=()=>{if(pending()||!b.recorded_model||R.norm(R.source(b,state))!==R.norm(b.source_text))return;R.modelResult(b,state,state.revision,R.source(b,state),b.recorded_model);persist();render();$('translationNotice').textContent='Loaded the recorded '+b.recorded_model.model+' response ('+b.recorded_model.seconds.toFixed(1)+' seconds in validation). No API call made. This draft is not independently reviewed.';};
  $('translateLive').onclick=async()=>{if(pending()||busy)return;const revision=state.revision,text=R.source(b,state);busy=true;statuses();$('translationNotice').textContent='Requesting English and Tamil drafts from Spark…';const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),55000);
    try{const res=await fetch('/api/reading/translate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text}),signal:controller.signal});const result=await res.json();if(!res.ok)throw Error(result.error||'Translation failed');if(result.source_text_sha256!==await hash(text))throw Error('Response does not match the requested Greek text');
      if(pending()||!R.modelResult(b,state,revision,text,result)){$('translationNotice').textContent='Reading changed while translating. The stale response was discarded; your edits are preserved.';}else{persist();render();$('translationNotice').textContent='New model drafts received. Review uncertain readings and both translations before treating them as reliable.';}}
    catch(e){$('translationNotice').textContent=e.name==='AbortError'?'Translation timed out. Your work is preserved.':e.message;}
    finally{clearTimeout(timeout);busy=false;statuses();}};
  function showRecognition(report,mode){const unsavedGreek=$('greekEditor').value;R.recognitionResult(b,state,report.prediction);persist();render();$('greekEditor').value=unsavedGreek;statuses();const e=report.evaluation;
    $('recognitionNotice').textContent=mode+' · '+report.prediction.model+' · '+report.prediction.seconds.toFixed(1)+' seconds. Select an image region to review its prediction. Existing corrections preserved.';
    $('recognitionMetrics').textContent='Published-reference comparison: normalized character error '+(e.normalized_cer*100).toFixed(1)+'%; strict character error '+(e.strict_cer*100).toFixed(1)+'%; normalized exact lines '+e.normalized_exact_lines+'/5. '+e.limitations;}
  $('recognizeRecorded').onclick=()=>{try{showRecognition(window.UNSCROLL_RECOGNITION,'Recorded image-only response; no API request');}catch(e){$('recognitionNotice').textContent=e.message;}};
  $('acceptPrediction').onclick=()=>{if(pending()||busy)return;R.acceptPrediction(b,state,selected);persist();render();$('editNotice').textContent='Prediction applied with provenance. Correct any errors and save; check translation status.';};
  $('recordedLocal').onclick=()=>{try{showRecognition(window.UNSCROLL_LOCAL_OCR,'Recorded CPU OCR; no inference run');}catch(e){$('recognitionNotice').textContent=e.message;}};
  async function runRecognition(local=false){if(busy)return;busy=true;statuses();$('recognitionNotice').textContent=local?'Recognizing five crops on CPU…':'Recognizing five image crops with Spark…';const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),55000);
    try{const res=await fetch(local?'/api/reading/recognize-local':'/api/reading/recognize',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}',signal:controller.signal});const report=await res.json();if(!res.ok)throw Error(report.error||'Recognition failed');
      // Preserve unsaved editor text too: loading predictions must never re-render it.
      const greek=$('greekEditor').value;showRecognition(report,local?'Live CPU OCR; no network inference':'Live image-only response');$('greekEditor').value=greek;statuses();
    }catch(e){$('recognitionNotice').textContent=e.name==='AbortError'?'Recognition timed out. Your edits are preserved.':e.message;}finally{clearTimeout(timeout);busy=false;statuses();}};
  $('recognizeLive').onclick=()=>runRecognition(false);
  $('recognizeLocal').onclick=()=>runRecognition(true);
  $('exportReading').onclick=()=>{if(pending())return;const review=R.exportReview(b,state);review.exported_at=new Date().toISOString();const url=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)],{type:'application/json;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='unscroll-reading-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  var STATIC = location.protocol === 'file:' || !/^(127\.0\.0\.1|localhost)$/.test(location.hostname);
  if(STATIC){for(const id of ['recognizeLive','recognizeLocal','translateLive']){const b=$(id);if(!b)continue;b.disabled=true;b.title='Needs the local server (Launch Unscroll.cmd): this page is served as static files.';}
    $('recognitionNotice').textContent='Live recognition needs the local server; the recorded results load here.';$('translationNotice').textContent='Live translation needs the local server; the recorded Spark draft loads here.';}
  render();if(state.recognition)$('recognitionNotice').textContent='Stored '+state.recognition.model+' image predictions restored. No new request made. Load recorded recognition or run again to display comparison metrics.';$('storageStatus').textContent='Local draft ready. No API request has been made by opening this page.';
})();
