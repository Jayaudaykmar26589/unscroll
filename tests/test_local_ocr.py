import hashlib,json,sys,types
from copy import deepcopy
from pathlib import Path
import pytest
from PIL import Image
from src.agents import local_ocr
from src.agents.reading import load_passage
from src.agents.recognition import recognize,contact_sheet

def test_contact_layout_preserves_region_order_and_bounds():
    crops=[Image.new('RGB',(10,8),'red'),Image.new('RGB',(6,9),'blue')]
    sheet=contact_sheet(crops)
    assert sheet.size==(180,194)
    assert sheet.getpixel((130,46))[0]>200
    assert sheet.getpixel((126,144))[2]>200
    assert contact_sheet(crops,True).size==sheet.size

@pytest.mark.parametrize('variant',['contact','gray_contact','gray_crops'])
def test_spark_transforms_never_send_reference_text(monkeypatch,variant):
    p=load_passage();seen=[]
    def provider(messages,timeout):
        seen.append(messages)
        return json.dumps({'lines':[{'id':f'r{i+1}','text':'?'} for i in range(5)]})
    recognize(provider,variant)
    for line in p['lines']:line['text']='SECRET_REFERENCE_SENTINEL'
    p['source_text']='SECRET_REFERENCE_SENTINEL'
    monkeypatch.setattr('src.agents.recognition.load_passage',lambda:p)
    recognize(provider,variant)
    assert seen[0]==seen[1]
    assert 'SECRET_REFERENCE_SENTINEL' not in json.dumps(seen)

def test_local_input_does_not_depend_on_reference_or_network(tmp_path,monkeypatch):
    p=load_passage();source=local_ocr.ROOT/'viewer'/p['source']['image_path']
    target=tmp_path/'viewer'/p['source']['image_path'];target.parent.mkdir(parents=True);target.write_bytes(source.read_bytes())
    modeldir=tmp_path/'tools/ocr/tessdata';modeldir.mkdir(parents=True)
    modeldir.joinpath('grc.traineddata').write_bytes(b'test-model')
    modeldir.joinpath('manifest.json').write_text(json.dumps({'sha256':hashlib.sha256(b'test-model').hexdigest()}))
    calls=[]
    class API:
        def __init__(self,**kwargs):assert kwargs['lang']=='grc' and kwargs['psm']==13
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def SetImage(self,im):calls.append((im.size,hashlib.sha256(im.tobytes()).hexdigest()))
        def GetUTF8Text(self):return 'α?'
    fake=types.SimpleNamespace(PyTessBaseAPI=API,OEM=types.SimpleNamespace(LSTM_ONLY=1),PSM=types.SimpleNamespace(RAW_LINE=13,SINGLE_LINE=7),tesseract_version=lambda:'test')
    monkeypatch.setitem(sys.modules,'tesserocr',fake)
    monkeypatch.setattr(local_ocr,'ROOT',tmp_path)
    monkeypatch.setattr(local_ocr,'load_passage',lambda:p)
    monkeypatch.setattr('src.agents.spark.call_spark',lambda *a,**k:pytest.fail('Local OCR must not use network model'))
    first=local_ocr.recognize_local(enhance=True,raw_line=True)
    for line in p['lines']:line['text']='REFERENCE MUST NOT ENTER OCR'
    second=local_ocr.recognize_local(enhance=True,raw_line=True)
    assert calls[:5]==calls[5:]
    assert first['lines']==second['lines']
    assert first['local_gpu_required'] is False
    assert first['lines'][0]['crop_size']==[1808,238]
    modeldir.joinpath('grc.traineddata').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='model hash'):local_ocr.recognize_local()

def test_local_endpoint_uses_cpu_engine_and_rejects_reference(monkeypatch):
    from tests.test_reading import request
    p=load_passage()
    def local(**kwargs):
        assert kwargs=={'enhance':True,'raw_line':True}
        return {'image_sha256':p['source']['image_sha256'],'lines':[{**l,'text':'?'} for l in p['lines']]}
    monkeypatch.setattr(local_ocr,'recognize_local',local)
    monkeypatch.setattr('src.agents.recognition.recognize',lambda:pytest.fail('Must not use Spark'))
    assert request(path='/api/reading/recognize-local',body={})[0]==200
    assert request(path='/api/reading/recognize-local',body={'reference':p['source_text']})[0]==400

def test_preprocessing_is_deterministic_and_does_not_change_source():
    im=Image.new('RGB',(30,10),(170,140,50));before=im.tobytes()
    a=local_ocr.prepare_crop(im,enhance=True)
    assert a.tobytes()==local_ocr.prepare_crop(im,enhance=True).tobytes()
    assert im.tobytes()==before and a.size==(108,68)
