"""Predeclared image-only ablation; save predictions before reading evaluation labels."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.agents.recognition import recognize,evaluate

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['baseline','contact','gray_contact','gray_crops','tesseract_raw','tesseract_gray','tesseract_binary','tesseract_red']);ap.add_argument('--raw-line',action='store_true');ap.add_argument('--run',type=int,default=1);a=ap.parse_args()
    out=ROOT/'reports/reading/experiments';out.mkdir(exist_ok=True)
    target=out/f'{a.variant}{"-rawline" if a.raw_line else ""}-{a.run}.json'
    if target.exists():raise SystemExit('Refusing to overwrite experiment')
    if a.variant.startswith('tesseract_'):
        from src.agents.local_ocr import recognize_local
        result=recognize_local(enhance=a.variant in ('tesseract_gray','tesseract_red'),binary=a.variant=='tesseract_binary',raw_line=a.raw_line,red=a.variant=='tesseract_red')
    else:result=recognize(variant=a.variant)
    target.write_text(json.dumps({'prediction':result},ensure_ascii=False,indent=2),encoding='utf-8')
    import psutil
    memory=psutil.Process().memory_info()
    report={'prediction':result,'evaluation':evaluate(result),'hardware':{'process_peak_working_set_bytes':getattr(memory,'peak_wset',memory.rss),'gpu_inference':False}}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'variant':a.variant,'seconds':result['seconds'],**report['evaluation']},ensure_ascii=True))
