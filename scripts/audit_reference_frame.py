"""Independent mesh/CT alignment evidence at the originally chosen reference.

No optimization over transforms. Compare documented XYZ with the erroneous XY
swap as a diagnostic, using physical plane intersections rather than vertices.
"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import tifffile
from PIL import Image,ImageDraw
from src.eval.meshref import load_obj,slice_segments,sample_curve,SurfaceIndex,validate_frame


def main():
    directory=ROOT/'data/cache/scroll1_segments';out=ROOT/'reports/mesh_reference';out.mkdir(parents=True,exist_ok=True)
    sid='20230518181521';v,f=load_obj(directory/(sid+'.obj'))
    volume=json.loads((directory/'volume.meta.json').read_text());meta=json.loads((directory/(sid+'.meta.json')).read_text())
    img=tifffile.memmap(ROOT/'data/cache/scroll1_6100_6200/06100.tif')
    transform=validate_frame(meta,volume,img.shape)
    seg=slice_segments(v,f,6100);points=np.concatenate([sample_curve(s)[0] for s in seg])
    profiles={}
    for label,q in [('documented_xyz',points),('incorrect_xy_swap',points[:,::-1])]:
        x=np.rint(q[:,0]).astype(int);y=np.rint(q[:,1]).astype(int)
        profiles[label]={'mean_ct':float(img[y,x].mean()),'median_ct':float(np.median(img[y,x]))}
    thumb=img[::7,::7].astype(float);lo,hi=np.percentile(thumb,[1,99]);im=Image.fromarray(np.uint8(np.clip((thumb-lo)/(hi-lo)*255,0,255))).convert('RGB')
    draw=ImageDraw.Draw(im)
    for s in seg:draw.line([tuple(p/7) for p in s],fill='#f697c6',width=3)
    p=np.array([4820,3403])/7;draw.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),outline='#7de0c2',width=3)
    draw.text((20,20),'Magenta: original reference mesh in documented XYZ; green: original click',fill='white')
    im.save(out/'coordinate_audit.png')
    result={'original_reference':sid,'volume_id':volume['uuid'],'mesh_to_volume':transform.tolist(),
            'mesh_bounds_xyz':[v.min(0).tolist(),v.max(0).tolist()],
            'click_xyz':[4820,3403,6100],
            'click_distance_to_original_mesh':float(SurfaceIndex(v,f).distances([[4820,3403,6100]])[0]),
            'ct_diagnostic_not_accuracy':profiles,
            'finding':'The reference shares the volume frame but occupies a different XY region. Z overlap alone was incorrectly treated as ground-truth coverage. No coordinate offset is warranted.'}
    (out/'coordinate_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
