"""Check published excerpt fidelity and saved browser export provenance."""
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agents.reading import load_passage,canonical_text,text_hash

class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)

def main():
    p=load_passage();html=Text();html.feed((ROOT/'data/reading/source.html').read_text(encoding='utf-8'))
    source=canonical_text(''.join(html.parts))
    checks={l['id']:canonical_text(l['text']) in source for l in p['lines']}
    if not all(checks.values()):raise ValueError('Excerpt differs from cached official publication: '+str(checks))
    image=ROOT/'viewer'/p['source']['image_path'];digest=hashlib.sha256(image.read_bytes()).hexdigest()
    if digest!=p['source']['image_sha256']:raise ValueError('Published image hash mismatch')
    result={'published_lines_exact_match':checks,'image_sha256':digest,'source_text_sha256':text_hash(p['source_text']),
            'source_html_sha256':hashlib.sha256((ROOT/'data/reading/source.html').read_bytes()).hexdigest(),
            'note':'Text fidelity and asset integrity checks, not independent OCR or translation accuracy.'}
    export=ROOT/'reports/reading/browser-export.json'
    if export.exists():
        e=json.loads(export.read_text(encoding='utf-8'));current=canonical_text('\n'.join(l['text'] for l in e['corrected_lines']))
        for target in ('en','ta'):
            t=e['translations'][target]
            if t['current']!=(canonical_text(t['source_text'])==current):raise ValueError('Export stale-state mismatch')
        if [l['text'] for l in e['published_lines']]!=[l['text'] for l in p['lines']]:raise ValueError('Export overwrote published reading')
        result['browser_export']={'revision':e['revision'],'history_events':len(e['history']),'published_source_preserved':True,'translation_source_binding_valid':True}
    (ROOT/'reports/reading/asset-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
