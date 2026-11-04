import io
import json
import subprocess
from email.message import Message
from pathlib import Path
import pytest
from src.agents.reading import load_passage,text_hash,translate_passage
from scripts.workbench import WorkbenchHandler,TRANSLATION_LOCK


def test_published_reading_preserves_uncertainty_and_image_regions():
    p=load_passage();assert len(p['lines'])==5
    assert '\u0323' in p['source_text']
    assert [l['id'] for l in p['lines']]==['l10','l11','l12','l13','l14']
    assert p['translations']['en']['source_text_sha256']==text_hash(p['source_text'])
    assert any('\u0b80'<=c<='\u0bff' for c in p['translations']['ta']['text'])
    assert 'not Unscroll OCR' in p['source']['transcription_status']


def test_source_hash_normalizes_unicode_but_not_changed_readings():
    assert text_hash('ά\r\nβ')==text_hash('α\u0301\nβ')
    assert text_hash('οὐκ')!=text_hash('οὐ?')


def test_translation_is_bound_to_exact_input_and_does_not_invent_confidence():
    seen=[]
    def provider(messages,timeout):
        seen.append(messages)
        return json.dumps({'en':'We do not think so.','ta':'நாம் அப்படி நினைப்பதில்லை.','notes':'Uncertain letters remain.'})
    result=translate_passage('οὐ̣κ νομίζομεν',provider)
    assert result['source_text_sha256']==text_hash('οὐ̣κ νομίζομεν')
    assert result['kind']=='model_draft' and 'conf' not in result
    assert seen[0][1]['content']=='οὐ̣κ νομίζομεν'


@pytest.mark.parametrize('raw',['not json','{}','{"en":"yes","ta":"Tamil-STUB","notes":"x"}','[]'])
def test_bad_provider_response_never_masquerades_as_translation(raw):
    with pytest.raises(ValueError,match='invalid English/Tamil'):
        translate_passage('λόγος',lambda *a,**k:raw)


@pytest.mark.parametrize('text',['','hello',None,'α'*4001])
def test_invalid_input_does_not_call_provider(text):
    def forbidden(*args,**kwargs):raise AssertionError('Provider must not run')
    with pytest.raises(ValueError):translate_passage(text,forbidden)


def request(origin='http://127.0.0.1:8765',body=None,content_type='application/json',path='/api/reading/translate'):
    h=WorkbenchHandler.__new__(WorkbenchHandler);h.path=path;h.headers=Message()
    raw=json.dumps(body if body is not None else {'text':'λόγος'}).encode()
    for k,v in {'Origin':origin,'Host':'127.0.0.1:8765','Content-Type':content_type,'Content-Length':str(len(raw))}.items():h.headers[k]=v
    h.rfile=io.BytesIO(raw);result=[];h.reply_json=lambda status,payload:result.append((status,payload))
    h.do_POST();return result[0]


def test_api_blocks_cross_origin_and_unknown_payload_before_provider(monkeypatch):
    monkeypatch.setattr('src.agents.reading.translate_passage',lambda *a:pytest.fail('must not call provider'))
    assert request(origin='https://attacker.example')[0]==403
    assert request(body={'text':'λόγος','key':'not-allowed'})[0]==400
    assert request(content_type='text/plain')[0]==415


def test_api_failure_keeps_key_details_private_and_releases_lock(monkeypatch):
    def fail(*a):raise RuntimeError('secret-provider-key')
    monkeypatch.setattr('src.agents.reading.translate_passage',fail)
    status,body=request();assert status==503 and 'secret' not in json.dumps(body)
    assert not TRANSLATION_LOCK.locked()


def test_api_returns_provenance_bound_success(monkeypatch):
    monkeypatch.setattr('src.agents.reading.translate_passage',lambda text:{'source_text_sha256':text_hash(text),'translations':{'en':'word','ta':'சொல்'}})
    status,body=request();assert status==200 and body['source_text_sha256']==text_hash('λόγος')


def test_browser_state_regressions():
    import shutil
    if shutil.which('node') is None:
        pytest.skip('Node.js not installed: browser-state tests in tests/reading_state_test.cjs need `node`')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['node',str(root/'tests/reading_state_test.cjs')],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
