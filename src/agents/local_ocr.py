"""CPU-only ancient-Greek OCR. No reference transcription enters Tesseract."""
import hashlib
import json
import time
from PIL import Image,ImageOps,ImageFilter
from src.agents.reading import ROOT,load_passage

def prepare_crop(image,enhance=False,binary=False,red=False):
    image=image.convert('RGB').getchannel('R') if red else ImageOps.grayscale(image)
    if binary:image=image.filter(ImageFilter.MedianFilter(3)).point(lambda p:0 if p<95 else 255)
    elif enhance:image=ImageOps.autocontrast(image.filter(ImageFilter.MedianFilter(3)),cutoff=1)
    image=image.resize((image.width*2,image.height*2),Image.Resampling.LANCZOS)
    return ImageOps.expand(image,border=24,fill=255)

def recognize_local(enhance=False,binary=False,raw_line=False,red=False):
    import tesserocr
    p=load_passage();path=ROOT/'viewer'/p['source']['image_path']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=p['source']['image_sha256']:raise ValueError('Source image changed')
    modeldir=ROOT/'tools/ocr/tessdata';model=modeldir/'grc.traineddata'
    modelhash=hashlib.sha256(model.read_bytes()).hexdigest()
    manifest=json.loads((modeldir/'manifest.json').read_text())
    if modelhash!=manifest['sha256']:raise ValueError('OCR model hash changed')
    start=time.perf_counter();rows=[]
    with Image.open(path) as image, tesserocr.PyTessBaseAPI(path=str(modeldir),lang='grc',oem=tesserocr.OEM.LSTM_ONLY,psm=tesserocr.PSM.RAW_LINE if raw_line else tesserocr.PSM.SINGLE_LINE) as api:
        for i,line in enumerate(p['lines']):
            crop=prepare_crop(image.crop(line['bbox']),enhance,binary,red)
            api.SetImage(crop)
            rows.append({'id':line['id'],'request_id':f'r{i+1}','bbox':line['bbox'],
                         'crop_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                         'crop_size':list(crop.size),'text':api.GetUTF8Text().strip()})
    result={'schema_version':1,'model':'Tesseract grc tessdata_best','engine':tesserocr.tesseract_version(),
            'model_sha256':modelhash,'kind':'image_prediction','image_sha256':digest,
            'seconds':time.perf_counter()-start,'preprocessing':'tesseract_red' if red else ('tesseract_binary' if binary else ('tesseract_gray' if enhance else 'tesseract_raw')),
            'raw_line':raw_line,
            'reference_sent':False,'local_gpu_required':False,'lines':rows}
    result['run_id']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result
