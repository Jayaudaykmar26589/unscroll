"""Fetch official CPU OCR weights and verify their pinned digest."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/grc.traineddata'
SHA='dfc9bda286cd9d8755b1832e5731a5425d1cf0803393a5fa6dee078466178cf1'
if __name__=='__main__':
    raw=urllib.request.urlopen(URL,timeout=45).read(10_000_001)
    if hashlib.sha256(raw).hexdigest()!=SHA:raise SystemExit('Unexpected model hash; download not installed')
    out=ROOT/'tools/ocr/tessdata';out.mkdir(parents=True,exist_ok=True)
    (out/'grc.traineddata').write_bytes(raw)
    (out/'manifest.json').write_text(json.dumps({'url':URL,'sha256':SHA,'bytes':len(raw)},indent=2))
    print('Verified ancient Greek model:',len(raw),'bytes')
