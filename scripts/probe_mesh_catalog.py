"""Cache public provenance metadata; no coordinate fitting."""
import json
import re
from pathlib import Path
import requests

BASE = 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/'
D = Path('data/cache/scroll1_segments')

def main():
    s = requests.Session()
    r = s.get(BASE+'paths/', timeout=40); r.raise_for_status()
    (D/'index.html').write_text(r.text)
    ids = re.findall(r'href="(\d{14})/', r.text)
    print('count', len(ids), ids, flush=True)
    for p in D.glob('*.obj'):
        r=s.get(BASE+'paths/'+p.stem+'/meta.json',timeout=30);r.raise_for_status()
        (D/(p.stem+'.meta.json')).write_text(r.text)
    r=s.get(BASE+'volumes/20230205180739/meta.json',timeout=30);r.raise_for_status()
    (D/'volume.meta.json').write_text(r.text)
    r=s.get(BASE+'paths/20230518181521/',timeout=30);r.raise_for_status()
    print(r.text[:16000])

if __name__ == '__main__': main()
