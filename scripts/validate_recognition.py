"""Run image recognition once, then score separately; reference never enters request."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agents.recognition import recognize,evaluate

if __name__=='__main__':
    result=recognize()
    report={'prediction':result,'evaluation':evaluate(result)}
    out=ROOT/'reports/reading/recognition-validation.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'viewer/recognition-data.js').write_text('window.UNSCROLL_RECOGNITION='+json.dumps(report,ensure_ascii=False)+';\n',encoding='utf-8')
    print(json.dumps({'seconds':result['seconds'],**report['evaluation']},ensure_ascii=True,indent=2))
