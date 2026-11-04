"""Replay the original 50 SAM masks through click-anchored CT ridge tracking.

Independent NCC compensates CT motion. Live LayerGuard thresholds are unchanged.
No new SAM inference or hand-picked per-slice corrections. Failed extraction is
kept empty, and neither a failed extraction nor a rejected guard can reanchor.
"""
import json
import hashlib
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from src.trace.sam3_tracker import postprocess_mask
from src.trace.guards import LayerGuard, load_cfg
from scripts.ridge_eval import ridge_mask,dilate2
from scripts.flatten_sheet import flatten
from scripts.validate_curve_fix import geometry,summarize,sheet_metrics


def comparison(images, old, new, out):
    both=np.concatenate(old+new)
    x0,y0=np.floor(both.min(0)-30).astype(int)
    x1,y1=np.ceil(both.max(0)+30).astype(int)
    w,h=x1-x0,y1-y0
    canvas=Image.new('RGB',(4*w*2,(h+25)*2),'#111')
    for frame in range(2):
        crop=images[frame][y0:y1,x0:x1].astype(float)
        lo,hi=np.percentile(crop,[1,99]); u=np.uint8(np.clip((crop-lo)/max(hi-lo,1)*255,0,255))
        for branch,points in enumerate((old[frame],new[frame])):
            im=Image.fromarray(u).convert('RGB').resize((w*2,h*2))
            draw=ImageDraw.Draw(im)
            draw.line([tuple((p-[x0,y0])*2) for p in points],fill='orange' if branch==0 else 'lime',width=2)
            x,y=(np.array([4820,3403])-[x0,y0])*2
            draw.ellipse((x-5,y-5,x+5,y+5),outline='red',width=2)
            col=frame*2+branch
            canvas.paste(im,(col*w*2,50))
            ImageDraw.Draw(canvas).text((col*w*2+4,8),f"{6100+frame}: {'medial axis' if branch==0 else 'CT ridge'}",fill='white')
    canvas.save(out/'image_comparison.png')


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='reports/departure_fix')
    args=ap.parse_args()
    out=ROOT/args.out; curves=out/'curves_corrected';curves.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'reports/curve_fix/curves_corrected'; baseline=ROOT/'reports/ridge_fix/curves_corrected'
    cfg=load_cfg(ROOT/'src/trace/guards.yaml');guard=LayerGuard(cfg)
    previous,accepted_image=None,None
    rows,first_images,first_old,first_new=[],[],[],[]
    paths=sorted(baseline.glob('[0-9]*.jsonl'))
    if len(paths)!=50:raise ValueError('need exactly 50 baseline slices')
    for frame,path in enumerate(paths):
        old=json.loads(path.read_text());z=old['source_z']
        with np.load(cache/f"mask_{old['z']:05d}.npz") as d: mask,tr=d['mask'],d['transform']
        xs,ys,ox,oy=tr
        gray=np.asarray(Image.open(cache/'frames3'/f'{frame:05d}.jpg'))[:,:,0]
        image=tifffile.imread(ROOT/'data/cache/scroll1_6100_6200'/old['source_file'])
        seed=((4820-ox)/xs,(3403-oy)/ys) if guard.reference is None else previous.mean(0)
        details={}
        p,raw=postprocess_mask(mask,gray,tr,None,seed=seed,previous=previous,
                               reference=guard.reference,reference_image=accepted_image,
                               image=image,details=details)
        if len(raw):previous=raw
        p=np.round(p,1)
        motion={**details.get('motion',{}),'reference_z':guard.reference_z}
        assessment=guard.evaluate(p,z,image_motion=motion)
        if guard.reference_z==z:accepted_image=image
        rec={**old,'points':p.tolist(),'origin':p[0].tolist() if len(p) else None,
             'extraction':details,'assessment':assessment}
        (curves/path.name).write_text(json.dumps(rec)+'\n')
        rm=dilate2(ridge_mask(image.astype(float)))
        pair={'z':z,'assessment':assessment,'extraction':details}
        for label,points in (('baseline',np.asarray(old['points'])),('corrected',p)):
            stats=geometry(points)
            if len(points):
                x=np.clip(points[:,0].astype(int),0,rm.shape[1]-1);y=np.clip(points[:,1].astype(int),0,rm.shape[0]-1)
                stats['ridge_frac2']=float(rm[y,x].mean())
            else: stats['ridge_frac2']=0.
            pair[label]=stats
        pair['mask_sha256']=hashlib.sha256((cache/f"mask_{old['z']:05d}.npz").read_bytes()).hexdigest()
        rows.append(pair)
        if frame<2:
            first_images.append(image);first_old.append(np.asarray(old['points']));first_new.append(p)
        m=details.get('motion',{})
        print(f"z={z} {assessment['status']} n={len(p)} {details['reason']} "
              f"motion=({m.get('dx')},{m.get('dy')}) NCC={m.get('score')} "
              f"maxstep={pair['corrected']['max_step']:.2f} ridge={pair['corrected']['ridge_frac2']:.3f}",flush=True)
    flatten(curves,ROOT/'data/cache/scroll1_6100_6200',out/'sheet_corrected')
    result={'method':'Original 50 cached SAM masks; click-anchored CT ridge DP and independent CT NCC; same live extraction and LayerGuard.',
            'limitations':'Evidence identifies medial-axis instability at 6101, not 20-voxel CT motion. ' 
            'Monotone ROI tracing is a local-sheet method, not a whole-scroll guarantee. ' 
            'Image consistency and geometric guards do not replace annotated layer IDs.',
            'guard_config':cfg,'summary':{},'guard_summary':{},'per_slice':rows}
    for label in ('baseline','corrected'):
        metrics=[r[label] for r in rows];stats=summarize(metrics)
        stats['mean_ridge_frac2']=float(np.mean([r['ridge_frac2'] for r in metrics]))
        stats['ridge_slices_ge_0_5']=sum(r['ridge_frac2']>=.5 for r in metrics)
        result['summary'][label]=stats
    result['guard_summary']={
        'ok':sum(r['assessment']['status']=='ok' for r in rows),
        'failed':sum(r['assessment']['status']=='failed' for r in rows),
        'suspect':sum(r['assessment']['status']=='suspect' for r in rows),
        'assessed':sum(r['assessment']['winding']['assessed'] for r in rows),
        'unassessed':sum(not r['assessment']['winding']['assessed'] for r in rows),
        'winding_alerts':sum(r['assessment']['winding']['pass'] is False for r in rows),
        'kink_failed_slices':sum(not r['assessment']['kink']['pass'] for r in rows)}
    result['flattening']=sheet_metrics(np.load(out/'sheet_corrected/sheet.npy'))
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    if all(len(p) for p in first_new):comparison(first_images,first_old,first_new,out)
    print(json.dumps({k:v for k,v in result.items() if k!='per_slice'},indent=2))


if __name__=='__main__':main()
