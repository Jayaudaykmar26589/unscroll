"""Serve only the public workbench directory on localhost; never the repo root."""
import argparse
import functools
import http.server
import sys
import webbrowser
import json
import threading
from urllib.parse import urlsplit
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

TRANSLATION_LOCK=threading.Lock()
PREDICT_LOCK=threading.Lock()

class WorkbenchHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Pages and data are rebuilt often; never let a browser show a stale copy.
        self.send_header('Cache-Control','no-store')
        super().end_headers()

    def reply_json(self,status,body):
        payload=json.dumps(body,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(payload)))
        self.end_headers();self.wfile.write(payload)

    def do_POST(self):
        if self.path=='/api/voice/transcribe':return self.voice()
        if self.path=='/api/predict/complete':return self.predict()
        if self.path not in ('/api/reading/translate','/api/reading/recognize','/api/reading/recognize-local'):return self.reply_json(404,{'error':'Unknown endpoint'})
        recognition=self.path!='/api/reading/translate'
        # A local page may call its own origin; unrelated websites may not spend API credits.
        origin=self.headers.get('Origin','');expected='http://'+self.headers.get('Host','')
        if origin!=expected or urlsplit(origin).hostname not in ('127.0.0.1','localhost'):
            return self.reply_json(403,{'error':'Same-origin local requests only'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply_json(415,{'error':'JSON required'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=20000:raise ValueError()
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict) or set(body)!=(set() if recognition else {'text'}):raise ValueError()
        except (ValueError,UnicodeDecodeError):return self.reply_json(400,{'error':'Invalid translation request'})
        if not TRANSLATION_LOCK.acquire(blocking=False):return self.reply_json(429,{'error':'A translation is already running'})
        try:
            if recognition:
                from src.agents.recognition import recognize,evaluate
                if self.path.endswith('/recognize-local'):
                    from src.agents.local_ocr import recognize_local
                    prediction=recognize_local(enhance=True,raw_line=True)
                else:prediction=recognize()
                self.reply_json(200,{'prediction':prediction,'evaluation':evaluate(prediction)})
            else:
                from src.agents.reading import translate_passage
                self.reply_json(200,translate_passage(body['text']))
        except ValueError as e:self.reply_json(422,{'error':str(e)})
        except Exception:
            # Provider errors can contain request details; never expose credentials or raw exceptions.
            self.reply_json(503,{'error':'Model service unavailable. Your corrections are preserved; use recorded drafts or edit manually.'})
        finally:TRANSLATION_LOCK.release()

    def predict(self):
        # Fragment -> whole sentence: corpus lookup (local), then Muse Spark only if nothing is attested.
        origin=self.headers.get('Origin','');expected='http://'+self.headers.get('Host','')
        if origin!=expected or urlsplit(origin).hostname not in ('127.0.0.1','localhost'):
            return self.reply_json(403,{'error':'Same-origin local requests only'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply_json(415,{'error':'JSON required'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8000:raise ValueError()
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict) or not isinstance(body.get('text'),str) or not set(body)<={'text','model','position'} or body.get('position','any') not in ('any','start','middle','end'):raise ValueError()
        except (ValueError,UnicodeDecodeError):return self.reply_json(400,{'error':'Invalid completion request'})
        from src.agents.complete import complete
        try:
            result=complete(body['text'],use_model=False)
            if result['status']!='not_found' or not body.get('model',True):return self.reply_json(200,result)
        except ValueError as e:return self.reply_json(422,{'error':str(e)})
        if not PREDICT_LOCK.acquire(blocking=False):return self.reply_json(429,{'error':'A prediction is already running'})
        try:self.reply_json(200,complete(body['text'],use_model=True,position=body.get('position','any')))
        except ValueError as e:self.reply_json(422,{'error':str(e)})
        except Exception:
            # Provider errors can contain request details; never expose credentials or raw exceptions.
            self.reply_json(503,{'error':'Muse Spark is unavailable. Attested lookup still works.'})
        finally:PREDICT_LOCK.release()

    def voice(self):
        # Push-to-talk clip -> Muse Voice Transcribe. Same-origin only; the key stays on this server.
        origin=self.headers.get('Origin','');expected='http://'+self.headers.get('Host','')
        if origin!=expected or urlsplit(origin).hostname not in ('127.0.0.1','localhost'):
            return self.reply_json(403,{'error':'Same-origin local requests only'})
        if self.headers.get('Content-Type','').split(';')[0]!='audio/wav':return self.reply_json(415,{'error':'audio/wav required'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 44<length<=5_000_000:raise ValueError()
            wav=self.rfile.read(length)
            if wav[:4]!=b'RIFF' or wav[8:12]!=b'WAVE':raise ValueError()
        except ValueError:return self.reply_json(400,{'error':'Invalid WAV clip (16 kHz mono 16-bit, at most 5 MB)'})
        try:
            from src.agents.voice import transcribe
            self.reply_json(200,{'transcript':transcribe(wav)})
        except Exception:
            # Provider errors can contain request details; never expose credentials or raw exceptions.
            self.reply_json(503,{'error':'Voice service unavailable. Use the keyboard: arrows, space.'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');ap.add_argument('--open',action='store_true');ap.add_argument('--port',type=int,default=8765)
    a=ap.parse_args()
    if a.build:
        from scripts.build_workbench import main as build
        build()
    if not (ROOT/'viewer/data.js').exists():raise SystemExit('Run scripts/build_workbench.py first')
    handler=functools.partial(WorkbenchHandler,directory=str(ROOT/'viewer'))
    server=http.server.ThreadingHTTPServer(('127.0.0.1',a.port),handler)
    url=f'http://127.0.0.1:{a.port}/home.html';print(url,flush=True)
    if a.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
