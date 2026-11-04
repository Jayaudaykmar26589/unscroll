"""Fetch the official published-passage page and list its illustration assets."""
import re
from pathlib import Path
import requests

def main():
    out=Path('data/reading');out.mkdir(parents=True,exist_ok=True)
    r=requests.get('https://scrollprize.org/grandprize',timeout=30);r.raise_for_status()
    (out/'source.html').write_text(r.text,encoding='utf-8')
    print('\n'.join(sorted(set(re.findall(r'(?:src|href)="([^"]+)',r.text)))))
    image='/img/grandprize/col-8.webp'
    if image not in r.text:raise ValueError('Published column image no longer linked by source')
    response=requests.get('https://scrollprize.org'+image,timeout=30);response.raise_for_status()
    if len(response.content)>2*1024*1024:raise ValueError('Unexpectedly large published image')
    Path('viewer/assets').mkdir(parents=True,exist_ok=True)
    Path('viewer/assets/published-col-8.webp').write_bytes(response.content)

if __name__=='__main__':main()
