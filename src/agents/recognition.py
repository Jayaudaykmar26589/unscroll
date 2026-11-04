"""Bounded image-only Greek recognition and reproducible reference scoring."""
import base64
import hashlib
import io
import json
import time
import unicodedata
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from src.agents import spark
from src.agents.reading import ROOT, load_passage, canonical_text

PROMPT = ('Transcribe visible ancient Greek in each supplied line crop, left to right. '
          'Do not reconstruct from known texts or infer missing words. Use ? for unreadable characters; '
          'empty text is allowed for an unreadable line. No translation or confidence scores. '
          'Return ONLY JSON {"lines":[{"id":"r1","text":"..."},...]}, one entry per supplied region.')

def contact_sheet(crops, grayscale=False):
    """Fixed image-only transform; no reference text, lexicon or learned restoration."""
    width=max(im.width for im in crops)+80
    height=sum(im.height+40 for im in crops)
    sheet=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(sheet)
    y=20
    for i,im in enumerate(crops):
        if grayscale:
            im=ImageOps.autocontrast(ImageOps.grayscale(im).filter(ImageFilter.MedianFilter(3)),cutoff=1).convert('RGB')
        draw.text((8,y+10),f'r{i+1}',fill='black')
        sheet.paste(im,(60,y));y+=im.height+40
    return sheet.resize((sheet.width*2,sheet.height*2),Image.Resampling.LANCZOS)

def recognize(call=None, variant='baseline', passage=None, timeout=45.):
    if variant not in ('baseline','contact','gray_contact','gray_crops'):raise ValueError('Unknown preprocessing variant')
    passage = passage or load_passage()
    image_path = ROOT/'viewer'/passage['source']['image_path']
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if digest != passage['source']['image_sha256']:
        raise ValueError('Source image changed; revalidate regions')
    content = []
    regions = [];crops=[]
    with Image.open(image_path) as image:
        for i, line in enumerate(passage['lines']):
            crop = image.crop(line['bbox']).convert('RGB')
            if variant=='gray_crops':
                from src.agents.local_ocr import prepare_crop
                crop=prepare_crop(crop,enhance=True).convert('RGB')
            crops.append(crop)
            buf = io.BytesIO(); crop.save(buf, format='PNG')
            raw = buf.getvalue(); rid = f'r{i+1}'
            regions.append({'id':line['id'], 'request_id':rid, 'bbox':line['bbox'],
                            'crop_sha256':hashlib.sha256(raw).hexdigest()})
            content.extend([{'type':'text','text':rid}, {'type':'image_url','image_url':{
                'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}])
    if variant in ('contact','gray_contact'):
        sheet=contact_sheet(crops,grayscale=variant=='gray_contact')
        buf=io.BytesIO();sheet.save(buf,format='PNG')
        content=[{'type':'text','text':'Each horizontal row is labeled r1 through r5. Transcribe only its adjacent crop.'},
                 {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}]
    if call is None:
        if spark.offline(): raise RuntimeError('Recognition requires API access; use recorded predictions')
        call = lambda messages, timeout: spark.call_spark(messages, timeout=timeout, max_retries=0)
    start = time.perf_counter()
    raw = call([{'role':'system','content':PROMPT}, {'role':'user','content':content}], timeout=timeout)
    try:
        obj = json.loads(raw)
        rows = obj['lines']
        expected = {r['request_id'] for r in regions}
        if not isinstance(rows,list) or len(rows)!=len(regions): raise ValueError()
        if any(not isinstance(r,dict) or set(r)!={'id','text'} or not isinstance(r['id'],str)
               or not isinstance(r['text'],str) or len(r['text'])>1000 for r in rows): raise ValueError()
        if {r['id'] for r in rows}!=expected: raise ValueError()
        by_id = {r['id']:canonical_text(r['text']) for r in rows}
    except (ValueError,TypeError,KeyError):
        raise ValueError('Recognition provider returned invalid region predictions') from None
    result = {'schema_version':1, 'model':spark.SPARK_MODEL, 'kind':'image_prediction',
              'image_sha256':digest, 'seconds':time.perf_counter()-start,
              'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
              'reference_sent':False, 'local_gpu_required':False,
              'preprocessing':variant,
              'request_images_sha256':[hashlib.sha256(base64.b64decode(c['image_url']['url'].split(',')[1])).hexdigest() for c in content if c['type']=='image_url'],
              'lines':[{**r,'text':by_id[r['request_id']]} for r in regions]}
    result['run_id'] = hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result

def distance(a,b):
    row = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt = [i]
        for j,y in enumerate(b,1): nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row = nxt
    return row[-1]

def normalized_letters(text):
    # Ignore scholarly accents, underdots, case, spacing/punctuation; retain ? as an error.
    text = unicodedata.normalize('NFD',text.lower()).replace('ϲ','σ').replace('ς','σ')
    return ''.join(c for c in text if unicodedata.category(c).startswith('L') or c=='?')

def evaluate(result, passage=None):
    p = passage or load_passage()
    if result['image_sha256']!=p['source']['image_sha256']: raise ValueError('Wrong reference image')
    rows = result['lines']
    if len(rows)!=len(p['lines']) or {r['id'] for r in rows}!={r['id'] for r in p['lines']}: raise ValueError('Wrong reference regions')
    predictions = {r['id']:r for r in rows}
    scores = []
    for ref in p['lines']:
        pred = predictions[ref['id']]
        if pred['bbox']!=ref['bbox']: raise ValueError('Wrong region coordinates')
        a,b = canonical_text(ref['text']),canonical_text(pred['text'])
        na,nb = normalized_letters(a),normalized_letters(b)
        scores.append({'id':ref['id'],'reference':a,'prediction':b,
                       'strict_edits':distance(a,b),'strict_characters':len(a),
                       'letter_edits':distance(na,nb),'reference_letters':len(na),
                       'normalized_exact':na==nb})
    return {'lines':scores,'strict_cer':sum(r['strict_edits'] for r in scores)/sum(r['strict_characters'] for r in scores),
            'normalized_cer':sum(r['letter_edits'] for r in scores)/sum(r['reference_letters'] for r in scores),
            'normalized_exact_lines':sum(r['normalized_exact'] for r in scores),
            'limitations':f"{len(scores)} manually located lines from one public image;" + ' not a held-out test. Model memorization cannot be excluded. Strict CER includes editorial marks; normalized CER removes diacritics, spaces and punctuation and folds sigma forms. No uncertainty-detection or scholarly accuracy claim.'}
