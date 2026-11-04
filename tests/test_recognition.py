import base64
import io
import json
import pytest
from PIL import Image
from src.agents.recognition import recognize,evaluate,distance,normalized_letters
from src.agents.reading import load_passage

def response():
    return {'lines':[{'id':f'r{i+1}','text':'?'} for i in range(5)]}

def test_request_contains_only_real_crops_and_anonymous_ids():
    p=load_passage()
    def provider(messages,timeout):
        assert timeout==45
        body=messages[1]['content']
        assert len(body)==10
        for i,line in enumerate(p['lines']):
            assert body[i*2]=={'type':'text','text':f'r{i+1}'}
            url=body[i*2+1]['image_url']['url']
            with Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))) as crop:
                x0,y0,x1,y1=line['bbox'];assert crop.size==(x1-x0,y1-y0)
            assert line['text'] not in json.dumps(messages,ensure_ascii=False)
        assert p['source']['url'] not in json.dumps(messages)
        return json.dumps(response())
    result=recognize(provider)
    assert result['reference_sent'] is False and result['local_gpu_required'] is False
    assert [r['id'] for r in result['lines']]==[r['id'] for r in p['lines']]

@pytest.mark.parametrize('raw',['bad','[]','{}',json.dumps({'lines':[{'id':'r1','text':'?'}]*5}),
    json.dumps({'lines':[{'id':f'r{i+1}','text':None} for i in range(5)]})])
def test_malformed_predictions_rejected(raw):
    with pytest.raises(ValueError,match='invalid region'):recognize(lambda *a,**k:raw)

def test_shuffled_predictions_are_mapped_by_id_not_position():
    obj={'lines':[{'id':f'r{i+1}','text':str(i)} for i in reversed(range(5))]}
    result=recognize(lambda *a,**k:json.dumps(obj))
    assert [r['text'] for r in result['lines']]==list('01234')

def test_evaluation_counts_unknowns_and_insertions_and_folds_only_documented_variants():
    assert distance('abc','axbc')==1
    assert distance('abc','')==3
    assert distance('','???')==3
    assert normalized_letters('Ϲάς σ̣')=='σασσ'
    assert normalized_letters('α?')=='α?'
    p=load_passage();r=recognize(lambda *a,**k:json.dumps(response()))
    for line,ref in zip(r['lines'],p['lines']):line['text']=ref['text']
    assert evaluate(r)['strict_cer']==0
    assert evaluate(r)['normalized_exact_lines']==5
    r['lines'][0]['text']+='?'*100
    assert evaluate(r)['normalized_cer']>1 # CER is not a bounded confidence score.
    r['lines'][0]['bbox']=[0,0,1,1]
    with pytest.raises(ValueError,match='coordinates'):evaluate(r)

def test_wrong_image_rejected_before_api(monkeypatch):
    p=load_passage();p['source']['image_sha256']='wrong'
    monkeypatch.setattr('src.agents.recognition.load_passage',lambda:p)
    with pytest.raises(ValueError,match='image changed'):
        recognize(lambda *a,**k:pytest.fail('provider must not run'))

def test_recognition_endpoint_rejects_client_text_and_paths(monkeypatch):
    from tests.test_reading import request
    monkeypatch.setattr('src.agents.recognition.recognize',lambda:pytest.fail('provider must not run'))
    assert request(path='/api/reading/recognize',body={'text':'answer'})[0]==400
    assert request(path='/api/reading/recognize',body={'path':'secret'})[0]==400
    assert request(path='/api/reading/recognize',body={},origin='https://outside.example')[0]==403

def test_recognition_endpoint_returns_scored_regions(monkeypatch):
    from tests.test_reading import request
    r=recognize(lambda *a,**k:json.dumps(response()))
    monkeypatch.setattr('src.agents.recognition.recognize',lambda:r)
    status,body=request(path='/api/reading/recognize',body={})
    assert status==200 and body['prediction']==r
    assert body['evaluation']['normalized_cer']>0
