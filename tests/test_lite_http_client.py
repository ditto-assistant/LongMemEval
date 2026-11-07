import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
seen = []

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == '/v1/chat/completions'
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert data['model'] in {'meta-llama/Meta-Llama-3.1-70B-Instruct', 'gpt-4o-2024-08-06'}
        if data['model'] == 'gpt-4o-2024-08-06':
            assert self.headers['Authorization'] == 'Bearer synthetic-only'
        else:
            assert self.headers['Authorization'] == 'Bearer EMPTY'
        content = data['messages'][0]['content']
        if 'synthetic_NO' not in content and 'synthetic_YES' not in content:
            self.send_error(400)
            return
        seen.append(content)
        label = 'no' if 'synthetic_NO' in content else 'yes'
        payload = json.dumps({'id':'synthetic', 'object':'chat.completion', 'created':0,
          'model':data['model'], 'choices':[{'index':0, 'message':{'role':'assistant','content':label}, 'finish_reason':'stop'}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *args): pass

def main():
    server = HTTPServer(('127.0.0.1', 0), Handler)
    base_url = f'http://127.0.0.1:{server.server_port}/v1'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            references = [{'question_id':f'q{i}', 'question_type':kind, 'question':f'Question {i}?', 'answer':f'Answer {i}'} for i, kind in enumerate(['single-session-user','single-session-preference','single-session-assistant','multi-session','temporal-reasoning','knowledge-update'])]
            references.append({'question_id':'q_abs','question_type':'single-session-user','question':'Unknown?','answer':'Not known'})
            hypotheses = [{'question_id':r['question_id'], 'hypothesis':'synthetic_NO' if i==2 else 'synthetic_YES'} for i,r in enumerate(references)]
            refs = tmp/'refs.json'; hyps = tmp/'hyps.jsonl'
            refs.write_text(json.dumps(references)); hyps.write_text(''.join(json.dumps(x)+'\n' for x in hypotheses))
            # Enforce loopback at the socket layer in the evaluator subprocess too.
            (tmp/'sitecustomize.py').write_text(
                "import socket\n"
                "original = socket.socket.connect\n"
                "def loopback(sock, address):\n"
                "    if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):\n"
                "        raise RuntimeError('non-loopback connection blocked')\n"
                "    return original(sock, address)\n"
                "socket.socket.connect = loopback\n"
            )
            env = {**os.environ, 'NO_PROXY':'*', 'no_proxy':'*', 'OPENAI_API_KEY':'synthetic-only',
                   'OPENAI_ORGANIZATION':'', 'OPENAI_ORG_ID':'', 'OPENAI_BASE_URL':base_url,
                   'LONGMEMEVAL_LOCAL_BASE_URL':base_url,
                   'PYTHONPATH':str(tmp)}
            for model in ('llama-3.1-70b-instruct', 'gpt-4o'):
                seen.clear()
                p = subprocess.run([sys.executable, str(repo/'src/evaluation/evaluate_qa.py'), model, str(hyps), str(refs)], env=env, capture_output=True, text=True, timeout=45)
                if p.returncode: raise RuntimeError(p.stderr[-3000:])
                rows = [json.loads(line) for line in Path(str(hyps)+'.eval-results-'+model).read_text().splitlines()]
                assert len(seen) == 7 and [r['autoeval_label']['label'] for r in rows] == [True,True,False,True,True,True,True]
                for i in range(6):
                    assert f'Question {i}?' in seen[i] and f'Answer {i}' in seen[i]
                assert 'Rubric:' in seen[1]
                assert 'off-by-one errors' in seen[4]
                assert 'updated answer' in seen[5]
                assert 'unanswerable question' in seen[6]
                assert 'Accuracy: 0.8571' in p.stdout
                print(f'{model}: real OpenAI SDK completed seven synthetic loopback requests')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
