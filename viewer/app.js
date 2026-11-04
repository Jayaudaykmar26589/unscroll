'use strict';
(() => {
  const data=window.UNSCROLL_DATA;
  if(!data){document.getElementById('verdict').textContent='Build the evidence bundle first';return;}
  const {report:r,frames}=data,$=id=>document.getElementById(id),pct=v=>(v*100).toFixed(1)+'%',num=v=>v.toFixed(2),N=frames.length;
  const key='unscroll-review-'+data.report_sha256;let marks=[],frame=0,timer=null,images=new Map();
  try{marks=JSON.parse(localStorage.getItem(key)||'[]');if(!Array.isArray(marks))marks=[];}catch{marks=[];$('storageNote').textContent='Local storage unavailable; export markers before closing.';}
  const b=r.summary.baseline,c=r.summary.corrected;
  $('accuracy').textContent=pct(c.within_2_voxels);$('distance').textContent=num(c.mean_voxels)+' vx';
  $('baselineDistance').textContent='Previous trace: '+num(b.mean_voxels)+' voxels';
  $('ridge').textContent=pct(r.proxy_summary.corrected.mean_ridge_frac2);
  $('guards').textContent=r.guard_summary.ok+' / '+N;
  // Near-click correspondence is evidence, not a semantic label certificate.
  $('verdict').textContent=c.within_2_voxels>=.9?'Strong local surface agreement':'Surface accuracy is not established';
  $('verdictNote').textContent=r.target_click_within_2_voxels
    ?'The click is near a released mesh. Inspect the full trace before treating continuity as correctness. This is a development benchmark.'
    :'The nearest catalog candidate is '+num(r.sources[0].manual_click_distance_voxels)+' voxels from the click. Target identity is unconfirmed; distances measure agreement with that fixed candidate.';
  $('reference').textContent='Fixed reference '+r.target_mesh+' · Volume '+r.volume_id+' · '+r.voxel_size_um+' µm/voxel';
  $('provenance').textContent=r.catalog_total+' catalog entries examined; '+r.catalog_unresolved.length+' unresolved. '+r.sources.length+' mesh candidates. No fitted coordinate offset. Report SHA-256: '+data.report_sha256.slice(0,20)+'…';
  $('slice').max=N-1;
  const [x0,y0]=r.roi_xyz[0],[x1,y1]=r.roi_xyz[1],ct=$('ct'),ctx=ct.getContext('2d');
  ct.width=(x1-x0)*2;ct.height=(y1-y0)*2;
  function save(){try{localStorage.setItem(key,JSON.stringify(marks));}catch{$('storageNote').textContent='Export markers to preserve them.';}renderMarks();}
  function renderMarks(){$('reviewCount').textContent=marks.length;$('marks').textContent=marks.filter(m=>m.source_z===frames[frame].z).length+' review markers on this slice';}
  function line(points,color,width=2){if(points.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>{const x=(p[0]-x0)*2,y=(p[1]-y0)*2;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  function drawCT(){ctx.fillStyle='#080e0e';ctx.fillRect(0,0,ct.width,ct.height);const f=frames[frame],im=images.get(frame);
    if(im?.complete&&im.naturalWidth)ctx.drawImage(im,0,0,ct.width,ct.height);
    if($('mesh').checked)r.per_slice[frame].reference_segments.forEach(s=>line(s,'#f697c6',3));
    if($('old').checked)line(f.baseline,'#efad69',2);
    if($('trace').checked)line(f.corrected,'#7de0c2',2);
    marks.filter(m=>m.source_z===f.z).forEach(m=>{ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc((m.x-x0)*2,(m.y-y0)*2,8,0,Math.PI*2);ctx.stroke();});
    ctx.fillStyle='#fff';ctx.font='16px Segoe UI';ctx.fillText('XYZ · full-resolution voxels',12,24);
  }
  function loadImage(i){if(images.has(i))return;const im=new Image();im.onload=()=>{if(frame===i)drawCT();};im.src=frames[i].image;images.set(i,im);
    // Bounded image cache: no volume or full-resolution slices in the browser.
    if(images.size>8){for(const k of images.keys()){if(Math.abs(k-frame)>2){images.delete(k);break;}}}}
  function drawChart(){const canvas=$('chart'),g=canvas.getContext('2d'),W=canvas.width,H=canvas.height,L=54,R=W-25,T=20,B=H-40;
    g.clearRect(0,0,W,H);const max=Math.max(2,...r.per_slice.flatMap(s=>[s.baseline.mean_voxels,s.corrected.mean_voxels]))*1.15;
    const X=i=>L+i/(N-1)*(R-L),Y=d=>B-d/max*(B-T);g.font='13px Segoe UI';
    for(let j=0;j<=3;j++){const v=max*j/3,y=Y(v);g.strokeStyle='#304043';g.beginPath();g.moveTo(L,y);g.lineTo(R,y);g.stroke();g.fillStyle='#a9b8b8';g.fillText(v.toFixed(0),8,y+4);}
    g.fillStyle='#a9b8b8';g.fillText('vx',10,14);g.fillText(frames[0].z,L,H-10);g.fillText(frames[N-1].z,R-35,H-10);
    g.setLineDash([4,5]);g.strokeStyle='#627b71';g.beginPath();g.moveTo(L,Y(2));g.lineTo(R,Y(2));g.stroke();g.setLineDash([]);
    for(const [branch,color] of [['baseline','#efad69'],['corrected','#7de0c2']]){g.strokeStyle=color;g.lineWidth=2;g.beginPath();r.per_slice.forEach((s,i)=>i?g.lineTo(X(i),Y(s[branch].mean_voxels)):g.moveTo(X(i),Y(s[branch].mean_voxels)));g.stroke();}
    g.strokeStyle='#fff';g.lineWidth=1;g.beginPath();g.moveTo(X(frame),T);g.lineTo(X(frame),B);g.stroke();
  }
  function drawSurface(){const canvas=$('surface'),g=canvas.getContext('2d'),W=canvas.width,H=canvas.height;g.clearRect(0,0,W,H);
    const project=(p,i)=>[W*.18+(p[1]-y0)/(y1-y0)*W*.65+i*1.5,H*.70-(p[0]-(x0+x1)/2)*.85-i*2];
    frames.forEach((f,i)=>{if(i%3&&i!==frame)return;g.strokeStyle=i===frame?'#ecede7':'#437e70';g.lineWidth=i===frame?2:1;g.beginPath();f.corrected.forEach((p,j)=>{const [x,y]=project(p,i);j?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();});
    g.fillStyle='#a9b8b8';g.font='12px Segoe UI';g.fillText('Oblique view · axial spacing exaggerated · selected slice in white',20,H-15);
  }
  function select(i){frame=Math.max(0,Math.min(N-1,Math.round(i)));$('slice').value=frame;const s=r.per_slice[frame],f=frames[frame];
    $('sliceLabel').textContent='z = '+f.z;$('sliceCount').textContent=(frame+1)+' / '+N;
    $('sliceAccuracy').textContent=pct(s.corrected.within_2_voxels);$('sliceDistance').textContent=num(s.corrected.mean_voxels)+' vx';$('sliceP95').textContent=num(s.corrected.p95_voxels)+' vx';$('sliceRidge').textContent=pct(s.corrected.ridge_frac2_proxy);$('sliceGuard').textContent=s.guard_status;$('sliceCoverage').textContent=s.reference_segments.length?'Present':'Absent';
    loadImage(frame);if(frame+1<N)loadImage(frame+1);drawCT();drawChart();drawSurface();renderMarks();}
  function togglePlay(){if(timer){clearInterval(timer);timer=null;$('play').textContent='▶';$('play').setAttribute('aria-label','Play slice sequence');}else{timer=setInterval(()=>select((frame+1)%N),180);$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Pause slice sequence');}}
  $('slice').oninput=e=>select(+e.target.value);$('play').onclick=togglePlay;
  for(const id of ['trace','old','mesh'])$(id).onchange=drawCT;
  $('worst').onclick=()=>select(r.per_slice.reduce((a,s,i)=>s.corrected.mean_voxels>r.per_slice[a].corrected.mean_voxels?i:a,0));
  $('chart').onclick=e=>{const rect=e.currentTarget.getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width*760;select((x-54)/(735-54)*(N-1));};
  document.addEventListener('keydown',e=>{if(['INPUT','BUTTON','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowRight'){e.preventDefault();select(frame+1);}if(e.key==='ArrowLeft'){e.preventDefault();select(frame-1);}if(e.code==='Space'){e.preventDefault();togglePlay();}});
  ct.onclick=e=>{if(!$('mark').checked)return;const box=ct.getBoundingClientRect();
    // Canvas has the same aspect ratio as its displayed box (CSS avoids letterboxing).
    marks.push({source_z:frames[frame].z,x:+(x0+(e.clientX-box.left)/box.width*(x1-x0)).toFixed(2),y:+(y0+(e.clientY-box.top)/box.height*(y1-y0)).toFixed(2),type:'review_requested'});save();drawCT();};
  function download(name,body,type='application/json'){const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('exportReport').onclick=()=>download('unscroll-surface-validation.json',JSON.stringify(r,null,2));
  $('exportMarks').onclick=()=>download('unscroll-review-markers.json',JSON.stringify({schema_version:1,report_sha256:data.report_sha256,volume_id:r.volume_id,coordinate_system:'volume voxel XYZ',markers:marks},null,2));
  $('clearMarks').onclick=()=>{marks=[];save();drawCT();};
  function resample(p,n){const arc=[0];for(let i=1;i<p.length;i++)arc.push(arc[i-1]+Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]));let j=0;return Array.from({length:n},(_,i)=>{const s=arc.at(-1)*i/(n-1);while(j<arc.length-2&&arc[j+1]<s)j++;const t=(s-arc[j])/Math.max(arc[j+1]-arc[j],1e-12);return p[j].map((v,k)=>v+t*(p[j+1][k]-v));});}
  $('exportMesh').onclick=()=>{const n=128,lines=['# Unscroll predicted surface; NOT released reference geometry','# Volume '+r.volume_id,'# XYZ voxels; arc-resampled rows, no reference fitting'];
    frames.forEach(f=>resample(f.corrected,n).forEach(p=>lines.push(`v ${p[0].toFixed(4)} ${p[1].toFixed(4)} ${f.z}`)));
    for(let i=0;i<N-1;i++)for(let j=0;j<n-1;j++){const a=i*n+j+1,b=a+1,c=a+n,d=c+1;lines.push(`f ${a} ${b} ${c}`,`f ${b} ${d} ${c}`);}
    download('unscroll-predicted-surface.obj',lines.join('\n')+'\n','text/plain');};
  select(0);
})();
