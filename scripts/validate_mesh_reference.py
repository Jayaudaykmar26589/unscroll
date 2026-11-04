"""CPU-only exact surface validation of the frozen 50-slice experiment.

No reference geometry enters extraction. Original and corrected traces share
the same fixed mesh selected once by the original manual click, never per slice.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import tifffile
import psutil
from src.eval.meshref import load_obj,validate_frame,curve_xyz,sample_curve,SurfaceIndex,slice_segments,paired_triangle_distance


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def metrics(dist,weights):
    if not len(dist):return {'assessed':False,'samples':0}
    order=np.argsort(dist);d=dist[order];w=weights[order];cs=np.cumsum(w)/w.sum()
    return {'assessed':True,'samples':len(d),'arc_length_voxels':float(w.sum()),
            'mean_voxels':float(np.average(d,weights=w)),
            'median_voxels':float(d[np.searchsorted(cs,.5)]),
            'p95_voxels':float(d[min(len(d)-1,np.searchsorted(cs,.95))]),
            'max_voxels':float(d.max()),
            'within_2_voxels':float(np.average(d<=2,weights=w)),
            'within_5_voxels':float(np.average(d<=5,weights=w))}


def clipped_segments(segments,roi):
    """Liang-Barsky clipping against fixed XY validation rectangle."""
    out=[]
    for a,b in segments:
        delta=b-a;lo,hi=0.,1.
        for axis in range(2):
            if abs(delta[axis])<1e-12:
                if not roi[0][axis]<=a[axis]<=roi[1][axis]:hi=-1;break
            else:
                t0=(roi[0][axis]-a[axis])/delta[axis];t1=(roi[1][axis]-a[axis])/delta[axis]
                lo=max(lo,min(t0,t1));hi=min(hi,max(t0,t1))
        if hi>lo:out.append([a+lo*delta,a+hi*delta])
    return np.asarray(out).reshape(-1,2,2)


def distance_to_curve(p,curve):
    if len(curve)<2:return np.full(len(p),np.inf)
    a=curve[:-1];edge=np.diff(curve,axis=0);den=np.maximum((edge*edge).sum(1),1e-20)
    out=[]
    for q in np.array_split(p,max(1,int(np.ceil(len(p)/128)))):
        delta=q[:,None,:]-a
        u=np.clip(np.einsum('nmd,md->nm',delta,edge)/den,0,1)
        out.extend(np.linalg.norm(delta-u[:,:,None]*edge,axis=2).min(1))
    return np.array(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='reports/mesh_reference');args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    directory=ROOT/'data/cache/scroll1_segments'
    catalog=json.loads((directory/'catalog.json').read_text());volume=json.loads((directory/'volume.meta.json').read_text())
    proxy_path=ROOT/'reports/departure_fix/verified/validation.json';proxy=json.loads(proxy_path.read_text())
    shape=tifffile.TiffFile(ROOT/'data/cache/scroll1_6100_6200/06100.tif').pages[0].shape
    sources=[];excluded=[];click=np.array([[4820,3403,6100.]])
    for row in catalog['meshes']:
        if not row.get('roi_overlap') or row.get('error'):continue
        sid=row['id'];path=directory/(sid+'.obj');meta=json.loads((directory/(sid+'.meta.json')).read_text())
        if meta.get('volume')!=volume['uuid']:
            excluded.append({'id':sid,'reason':'different_volume','volume':meta.get('volume')})
            continue
        validate_frame(meta,volume,shape)
        v,f=load_obj(path)
        # One seed query does not justify building a BVH for every catalog mesh.
        seed_distance=min(float(paired_triangle_distance(np.broadcast_to(click,(len(ids),3)),v[ids]).min())
                          for ids in np.array_split(f,max(1,int(np.ceil(len(f)/4096)))))
        sources.append({'id':sid,'url':row['url'],'sha256':sha(path),'volume':meta['volume'],
                        'metadata_sha256':sha(directory/(sid+'.meta.json')),
                        'bounds_xyz':[v.min(0).tolist(),v.max(0).tolist()],
                        'manual_click_distance_voxels':seed_distance})
        print('reference',sid,'click distance',seed_distance,flush=True)
    if not sources:raise ValueError('No mesh candidates; run catalog_meshes.py')
    sources.sort(key=lambda r:r['manual_click_distance_voxels']);target=sources[0]['id']
    v,f=load_obj(directory/(target+'.obj'));index=SurfaceIndex(v,f)
    branches={'baseline':ROOT/'reports/ridge_fix/curves_corrected','corrected':ROOT/'reports/departure_fix/verified/curves_corrected'}
    aggregate={k:([],[]) for k in branches};rows=[];hashes=[];refds=[];refws=[]
    peak=0
    for frame in range(50):
        z=6100+frame;item={'source_z':z};raw={}
        segments=clipped_segments(slice_segments(v,f,z),catalog['roi_xyz'])
        for label,folder in branches.items():
            path=folder/f'{frame:05d}.jsonl';rec=json.loads(path.read_text());p=curve_xyz(rec)
            if rec['source_z']!=z:raise ValueError('Unexpected physical slice')
            q,w=sample_curve(p);d=index.distances(q)
            item[label]=metrics(d,w);item[label]['ridge_frac2_proxy']=proxy['per_slice'][frame][label]['ridge_frac2']
            aggregate[label][0].extend(d);aggregate[label][1].extend(w);raw[label]=p[:,:2]
            hashes.append({'branch':label,'file':str(path.relative_to(ROOT)),'sha256':sha(path)})
        # Fixed-ROI reference -> trace coverage. This is slice-length coverage,
        # not whole-mesh area recall and not a complete annotation of all layers.
        rp=[];rw=[]
        for seg in segments:
            q,w=sample_curve(seg);rp.extend(q);rw.extend(w)
        if rp:
            rd=distance_to_curve(np.array(rp),raw['corrected']);refds.extend(rd);refws.extend(rw)
            item['reference_to_trace_in_roi']=metrics(rd,np.array(rw))
        else:item['reference_to_trace_in_roi']={'assessed':False,'samples':0}
        item['reference_segments']=segments.round(4).tolist()
        item['guard_status']=proxy['per_slice'][frame]['assessment']['status']
        rows.append(item);peak=max(peak,psutil.Process().memory_info().rss)
        print(z,'mean',round(item['corrected']['mean_voxels'],3),'within2',round(item['corrected']['within_2_voxels'],3),flush=True)
    result={'schema_version':1,'volume_id':volume['uuid'],'voxel_size_um':volume['voxelsize'],
            'coordinate_contract':'OBJ XYZ = TIFF column,row,physical source_z; identity transform; no fitted translation, permutation or registration.',
            'target_mesh':target,'selection':'Nearest mesh to the original manual click, selected once and fixed across both branches and all 50 slices.',
            'target_identity_confirmed':False,
            'target_click_within_2_voxels':sources[0]['manual_click_distance_voxels']<=2,
            'sources':sources,'excluded_sources':excluded,'catalog_sha256':sha(directory/'catalog.json'),
            'volume_metadata_sha256':sha(directory/'volume.meta.json'),
            'catalog_total':len(catalog['meshes']),
            'catalog_unresolved':[r['id'] for r in catalog['meshes'] if not r.get('complete') or r.get('error')],
            'roi_xyz':catalog['roi_xyz'],'summary':{k:metrics(np.array(d),np.array(w)) for k,(d,w) in aggregate.items()},
            'reference_to_trace_in_roi':metrics(np.array(refds),np.array(refws)),
            'proxy_summary':proxy['summary'],'guard_summary':proxy['guard_summary'],
            'proxy_source':{'file':str(proxy_path.relative_to(ROOT)),'sha256':sha(proxy_path)},
            'trace_sources':hashes,'per_slice':rows,
            'limitations':['Released meshes are partial surface annotations, not exhaustive layer ground truth.',
                          'Distance to a released surface is actual geometric agreement, not proof of target layer identity.',
                          'Coverage is length-weighted within the fixed ROI and 50 slice planes, not whole-mesh area recall.',
                          'Ridge and guards are reused frozen proxy results; no new SAM inference or mesh-guided tracing.',
                          'This previously tuned patch is a development benchmark, not a held-out accuracy estimate.'],
            'runtime':{'seconds':time.perf_counter()-start,'peak_sampled_rss_mb':peak/1048576,
                       'os_peak_rss_mb':getattr(psutil.Process().memory_info(),'peak_wset',peak)/1048576,
                       'gpu_used':False}}
    (out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('target_mesh','target_identity_confirmed','summary','reference_to_trace_in_roi','runtime')},indent=2))

if __name__=='__main__':main()
