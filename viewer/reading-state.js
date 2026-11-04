/* Shared, testable state rules; published evidence remains immutable. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.ReadingState=api;})(typeof window==='object'?window:globalThis,function(){
  const norm=s=>String(s).replace(/\r\n?/g,'\n').normalize('NFC').trim();
  const source=(bundle,state)=>bundle.lines.map(l=>state.lines[l.id]).join('\n');
  function create(bundle){return {bundle_id:bundle.bundle_id,revision:0,lines:Object.fromEntries(bundle.lines.map(l=>[l.id,l.text])),history:[],translations:Object.fromEntries(['en','ta'].map(k=>[k,{text:bundle.translations[k].text,source_text:norm(bundle.source_text),kind:bundle.translations[k].kind||'editorial_draft',notes:bundle.translations[k].review_status}]))};}
  function restore(bundle,saved){if(!saved||saved.bundle_id!==bundle.bundle_id)return create(bundle);const s=create(bundle);
    if(!Number.isSafeInteger(saved.revision)||saved.revision<0||!Array.isArray(saved.history))return s;
    for(const l of bundle.lines)if(typeof saved.lines?.[l.id]!=='string'||saved.lines[l.id].length>1000)return s;
    for(const k of ['en','ta']){const t=saved.translations?.[k];if(!t||!['text','source_text','kind','notes'].every(p=>typeof t[p]==='string')||t.text.length>12000)return s;}
    const restored=JSON.parse(JSON.stringify(saved));if(restored.recognition){try{recognitionResult(bundle,restored,restored.recognition);}catch{delete restored.recognition;}}return restored;}
  function edit(bundle,s,id,text){if(!bundle.lines.some(l=>l.id===id)||typeof text!=='string'||text.length>1000)throw Error('Invalid line correction');if(s.lines[id]===text)return false;
    s.history.push({type:'transcription',line_id:id,before:s.lines[id],after:text,revision:++s.revision,at:new Date().toISOString()});s.lines[id]=text;return true;}
  function current(bundle,s,target){return norm(s.translations[target].source_text)===norm(source(bundle,s));}
  function humanTranslation(bundle,s,target,text){if(!['en','ta'].includes(target)||!text.trim()||text.length>12000)throw Error('Enter a translation');
    s.history.push({type:'translation',target,before:s.translations[target].text,after:text,revision:++s.revision,at:new Date().toISOString()});
    s.translations[target]={text,source_text:norm(source(bundle,s)),kind:'human_draft',notes:'User-edited draft; no independent scholarly review recorded.'};}
  function modelResult(bundle,s,requestRevision,requestText,result){if(s.revision!==requestRevision||norm(source(bundle,s))!==norm(requestText))return false;
    if(!result?.translations||!['en','ta'].every(k=>typeof result.translations[k]==='string'&&result.translations[k].trim()))throw Error('Incomplete model result');
    for(const k of ['en','ta'])s.translations[k]={text:result.translations[k],source_text:norm(requestText),kind:'model_draft',notes:result.notes||'Unreviewed model draft'};
    s.history.push({type:'model_translation',revision:++s.revision,model:result.model,source_text_sha256:result.source_text_sha256,at:new Date().toISOString()});return true;}
  function recognitionResult(bundle,s,result){
    if(result?.image_sha256!==bundle.source.image_sha256||typeof result.run_id!=='string'||!Array.isArray(result.lines)||result.lines.length!==bundle.lines.length)throw Error('Prediction image or regions do not match');
    const ids=new Set();for(const row of result.lines){const ref=bundle.lines.find(l=>l.id===row.id);if(!ref||ids.has(row.id)||JSON.stringify(row.bbox)!==JSON.stringify(ref.bbox)||typeof row.text!=='string'||row.text.length>1000)throw Error('Invalid prediction region');ids.add(row.id);}
    s.recognition=JSON.parse(JSON.stringify(result));
  }
  function acceptPrediction(bundle,s,id){const row=s.recognition?.lines.find(l=>l.id===id);if(!row)throw Error('No prediction for this region');
    const before=s.lines[id];s.lines[id]=row.text;s.history.push({type:'recognition_accept',line_id:id,before,after:row.text,run_id:s.recognition.run_id,model:s.recognition.model,revision:++s.revision,at:new Date().toISOString()});}
  function exportReview(bundle,s){return {schema_version:2,bundle_id:bundle.bundle_id,source:bundle.source,source_text_sha256:bundle.source_text_sha256,published_lines:bundle.lines,recognition:s.recognition||null,corrected_lines:bundle.lines.map(l=>({...l,text:s.lines[l.id],changed:s.lines[l.id]!==l.text})),translations:Object.fromEntries(['en','ta'].map(k=>[k,{...s.translations[k],current:current(bundle,s,k)}])),history:s.history,revision:s.revision,disclosure:bundle.source.relationship_to_experiment};}
  return {norm,source,create,restore,edit,current,humanTranslation,modelResult,recognitionResult,acceptPrediction,exportReview};
});
