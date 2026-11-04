"""Bounded public OBJ catalog scan. Save only meshes intersecting a fixed ROI.

Stops vertex scan at faces (VolCart writes vertices first). Oversized files are
explicitly unresolved, never recorded as non-overlapping. No trace fitting.
"""
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
import requests
import numpy as np

BASE='https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/'
D=Path('data/cache/scroll1_segments')
ROI=np.array([[4700,3150,6100],[4950,3650,6149]])
LIMIT=64*1024*1024

def probe(sid):
    row={'id':sid,'url':BASE+sid+'/'+sid+'.obj'}
    try:
        lo=np.full(3,np.inf);hi=-lo.copy();n=0;size=0;complete=False
        with requests.get(row['url'],stream=True,timeout=45) as r:
            r.raise_for_status()
            for line in r.iter_lines(chunk_size=65536):
                size+=len(line)+1
                if size>LIMIT:break
                if line.startswith(b'f '):complete=True;break
                if line.startswith(b'v '):
                    v=np.array([float(x) for x in line.split()[1:4]])
                    lo=np.minimum(lo,v);hi=np.maximum(hi,v);n+=1
        row.update(bounds=[lo.tolist(),hi.tolist()],vertices=n,scanned_bytes=size,complete=complete)
        overlap=complete and n>0 and np.all(hi>=ROI[0]) and np.all(lo<=ROI[1])
        row['roi_overlap']=bool(overlap)
        if overlap:
            with requests.get(row['url'],stream=True,timeout=60) as r:
                r.raise_for_status();total=0
                p=D/(sid+'.obj.part')
                with p.open('wb') as f:
                    for chunk in r.iter_content(65536):
                        total+=len(chunk)
                        if total>256*1024*1024:raise ValueError('mesh exceeds 256 MiB cap')
                        f.write(chunk)
                p.replace(D/(sid+'.obj'))
            row['sha256']=hashlib.sha256((D/(sid+'.obj')).read_bytes()).hexdigest()
            r=requests.get(BASE+sid+'/meta.json',timeout=30);r.raise_for_status()
            (D/(sid+'.meta.json')).write_text(r.text)
            print('MATCH',sid,lo,hi,flush=True)
    except Exception as e:row['error']=str(e)
    return row

def main():
    ids=re.findall(r'href="(\d{14})/',(D/'index.html').read_text())
    path=D/'catalog.json';prior=json.loads(path.read_text())['meshes'] if path.exists() else []
    done={r['id']:r for r in prior if r.get('complete') and not r.get('error')}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for row in ex.map(probe,[s for s in ids if s not in done]):
            done[row['id']]=row
            path.write_text(json.dumps({'roi_xyz':ROI.tolist(),'meshes':list(done.values())},indent=2))
            print(len(done),row['id'],row.get('complete'),row.get('error',''),flush=True)

if __name__=='__main__':main()
