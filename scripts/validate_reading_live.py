"""One explicit live translation probe; no keys or raw provider errors logged."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agents.reading import load_passage,translate_passage

def main():
    passage=load_passage();start=time.perf_counter()
    try:
        result=translate_passage(passage['source_text'])
        report={'status':'success','seconds':time.perf_counter()-start,'input':passage['source_text'],**result}
    except Exception as e:
        report={'status':'unavailable','seconds':time.perf_counter()-start,'error_type':type(e).__name__,
                'note':'No model result claimed; editorial drafts remain available offline.'}
    out=ROOT/'reports/reading';out.mkdir(parents=True,exist_ok=True)
    (out/'live-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
    return 0 if report['status']=='success' else 1

if __name__=='__main__':raise SystemExit(main())
