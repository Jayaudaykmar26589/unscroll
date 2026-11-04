"""Build a small, self-contained CPU-only evidence viewer from validated files."""
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import tifffile
from PIL import Image
from scripts.validate_mesh_reference import sha


def verify_sources(root,report):
    # Refuse to display stale evaluation next to modified geometry.
    for entry in report['trace_sources']:
        if sha(root/entry['file'])!=entry['sha256']:raise ValueError('Trace changed since evaluation: '+entry['file'])
    if sha(root/report['proxy_source']['file'])!=report['proxy_source']['sha256']:
        raise ValueError('Proxy report changed since evaluation')


def main():
    report_path=ROOT/'reports/mesh_reference/validation.json'
    report=json.loads(report_path.read_text())
    verify_sources(ROOT,report)
    out=ROOT/'viewer';assets=out/'assets';assets.mkdir(parents=True,exist_ok=True)
    roi=report['roi_xyz'];x0,y0=roi[0][:2];x1,y1=roi[1][:2]
    frames=[]
    for i,row in enumerate(report['per_slice']):
        z=row['source_z'];path=ROOT/f'data/cache/scroll1_6100_6200/{z:05d}.tif'
        # Original TIFFs are uncompressed; map the crop without allocating a volume.
        mapped=tifffile.memmap(path);crop=np.asarray(mapped[y0:y1,x0:x1],float)
        lo,hi=np.percentile(crop,[1,99]);pixels=np.uint8(np.clip((crop-lo)/max(hi-lo,1)*255,0,255))
        Image.fromarray(pixels).save(assets/f'{z}.webp',lossless=True)
        del mapped
        record={'z':z,'image':f'assets/{z}.webp'}
        for label,folder in (('baseline','reports/ridge_fix/curves_corrected'),('corrected','reports/departure_fix/verified/curves_corrected')):
            record[label]=json.loads((ROOT/folder/f'{i:05d}.jsonl').read_text())['points']
        frames.append(record)
    shutil.copyfile(ROOT/'reports/departure_fix/verified/sheet_corrected/flat_preview.png',assets/'flat.png')
    shutil.copyfile(report_path,out/'validation.json')
    bundle={'report':report,'frames':frames,'report_sha256':sha(report_path)}
    (out/'data.js').write_text('window.UNSCROLL_DATA='+json.dumps(bundle,allow_nan=False)+';\n',encoding='utf-8')
    from scripts.build_reading import main as build_reading
    build_reading()
    print(f'Built {out}/index.html: {len(frames)} slices, no GPU or API required')

if __name__=='__main__':main()
