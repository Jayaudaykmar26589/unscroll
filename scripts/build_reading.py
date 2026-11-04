"""Validate and bundle a published reading example; no API/GPU required."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from src.agents.reading import load_passage,text_hash

def main():
    data=load_passage();image=ROOT/'viewer'/data['source']['image_path']
    with Image.open(image) as im:
        if im.size!=(data['source']['image_width'],data['source']['image_height']):raise ValueError('Published image dimensions changed')
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    if digest!=data['source']['image_sha256']:raise ValueError('Published image hash changed; recheck line coordinates before rebuilding')
    recorded=ROOT/'reports/reading/live-validation.json'
    if recorded.exists():
        response=json.loads(recorded.read_text(encoding='utf-8'))
        if response.get('status')=='success' and response.get('source_text_sha256')==text_hash(data['source_text']):
            data['recorded_model']=response
    data['recognition_supported']=True       # image-recognition runs exist for this excerpt only
    data['bundle_id']=hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    bundles=[data]
    for path in sorted((ROOT/'data/reading/passages').glob('*.json'),key=lambda p:int(p.stem.rsplit('minus',1)[1]),reverse=True):
        extra=load_passage(path);extra.pop('translation_input',None)
        im=ROOT/'viewer'/extra['source']['image_path']
        if hashlib.sha256(im.read_bytes()).hexdigest()!=extra['source']['image_sha256']:raise ValueError(f'{path.name}: image changed; recheck line rectangles')
        extra['recognition_supported']=False
        cr=ROOT/'reports/reading/columns'/f"{extra['id']}.json"
        if cr.exists():                      # scripts/read_columns.py: Spark reads this column, scored
            r=json.loads(cr.read_text(encoding='utf-8'))
            if r['prediction']['image_sha256']==extra['source']['image_sha256']:
                e=r['evaluation'];extra['column_reading']={'model':r['prediction']['model'],'seconds':round(r['prediction']['seconds'],1),
                    'lines':len(e['lines']),'exact_lines':e['normalized_exact_lines'],'normalized_cer':e['normalized_cer']}
        extra['bundle_id']=hashlib.sha256(json.dumps(extra,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        bundles.append(extra)
    fp=None
    if (ROOT/'reports/ink_gp_full/reading.json').exists():  # scripts/read_fullpage.py: full-page ink map + Spark lines
        fp=json.loads((ROOT/'reports/ink_gp_full/reading.json').read_text(encoding='utf-8'))
        fp['ink']=json.loads((ROOT/'reports/ink_gp_full/summary.json').read_text())
        cs=ROOT/'reports/reading/columns/summary.json'
        if cs.exists():fp['published_column_cer']=json.loads(cs.read_text())['overall_normalized_cer']
    # ?p=<id> picks a passage; the first is the original excerpt
    (ROOT/'viewer/reading-data.js').write_text('window.UNSCROLL_READINGS='+json.dumps(bundles,ensure_ascii=False)+';\n'
        "window.UNSCROLL_READING=(function(){var q=new URLSearchParams(location.search).get('p');"
        "return UNSCROLL_READINGS.filter(function(b){return b.id===q;})[0]||UNSCROLL_READINGS[0];})();\n"
        'window.UNSCROLL_FULLPAGE='+json.dumps(fp,ensure_ascii=False)+';\n',encoding='utf-8')
    out=ROOT/'reports/reading';out.mkdir(parents=True,exist_ok=True)
    (out/'source-manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    recognition=out/'recognition-validation.json'
    report=None
    if recognition.exists():
        from src.agents.recognition import evaluate
        report=json.loads(recognition.read_text(encoding='utf-8'))
        report['evaluation']=evaluate(report['prediction'],data)
    (ROOT/'viewer/recognition-data.js').write_text('window.UNSCROLL_RECOGNITION='+json.dumps(report,ensure_ascii=False)+';\n',encoding='utf-8')
    from src.agents.recognition import evaluate
    local=out/'experiments/tesseract_gray-rawline-2.json'
    local_report=None
    if local.exists():
        local_report=json.loads(local.read_text(encoding='utf-8'))
        local_report['evaluation']=evaluate(local_report['prediction'],data)
    with (ROOT/'viewer/recognition-data.js').open('a',encoding='utf-8') as f:
        f.write('window.UNSCROLL_LOCAL_OCR='+json.dumps(local_report,ensure_ascii=False)+';\n')
    print('Published passages validated:',', '.join(f"{b['id']} ({len(b['lines'])} lines)" for b in bundles))

if __name__=='__main__':main()
