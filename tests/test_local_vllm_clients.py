"""Regression tests for local vLLM credentials and HTTP transport."""

import contextlib
import ast
import http.server
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import openai

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_KEY = "local-test-key"


@contextlib.contextmanager
def server(response=200, location=None):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Authorization") == "Bearer " + SYNTHETIC_KEY)
            self.send_response(response)
            if location:
                self.send_header("Location", location)
            self.end_headers()
            self.wfile.write(b'{}')

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()


class LocalClientsTest(unittest.TestCase):
    def test_retrieval_client_construction(self):
        source = (ROOT / "src/retrieval/run_retrieval.py").read_text()
        tree = ast.parse(source)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "client" for t in node.targets))
        code = compile(ast.Module(body=[assignment], type_ignores=[]), "run_retrieval.py", "exec")
        for host, expected in (("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")):
            calls = []
            def capture(**kwargs):
                calls.append(kwargs)
            with patch.dict(os.environ, {"VLLM_HOST": host, "VLLM_API_KEY": SYNTHETIC_KEY}):
                exec(code, {"os": os, "OpenAI": capture, "DefaultHttpxClient": openai.DefaultHttpxClient})
            self.assertEqual(calls[0]["api_key"], SYNTHETIC_KEY)
            self.assertIn(expected, calls[0]["base_url"])
            self.assertFalse(calls[0]["http_client"].trust_env)
            self.assertFalse(calls[0]["http_client"].follow_redirects)
            calls[0]["http_client"].close()

    def test_wrapper_does_not_put_environment_key_in_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script_dir = root / "src/generation"
            script_dir.mkdir(parents=True)
            shutil.copy(ROOT / "src/generation/run_generation.sh", script_dir)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_python = bin_dir / "python"
            fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
            fake_python.chmod(0o755)
            for host, expected in (("127.0.0.1", "http://127.0.0.1:8001/v1"), ("::1", "http://[::1]:8001/v1")):
                capture = root / "args"
                env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], CAPTURE_ARGS=str(capture), VLLM_HOST=host, VLLM_API_KEY=SYNTHETIC_KEY)
                subprocess.run(['bash', 'run_generation.sh', 'input.json', 'llama-3.1-8b-instruct', 'no-retrieval'], cwd=script_dir, env=env, check=True, capture_output=True)
                args = capture.read_text().splitlines()
                self.assertIn('--local_vllm', args)
                self.assertIn(expected, args)
                self.assertNotIn(SYNTHETIC_KEY, args)
            subprocess.run(['bash', 'run_generation.sh', 'input.json', 'gpt-4o', 'no-retrieval'], cwd=script_dir, env=env, check=True, capture_output=True)
            args = capture.read_text().splitlines()
            self.assertNotIn('--local_vllm', args)
            self.assertNotIn(SYNTHETIC_KEY, args)

    def test_expansion_clients_use_local_key_and_safe_transport(self):
        paths = sorted((ROOT / "src/index_expansion").glob("batch_expansion_*.py"))
        self.assertEqual(len(paths), 6)
        for host, expected in (("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")):
            for key in (SYNTHETIC_KEY, ""):
                for path in paths:
                    with self.subTest(host=host, keyed=bool(key), script=path.name):
                        calls = []
                        def capture(**kwargs):
                            calls.append(kwargs)
                            return object()
                        with patch.dict(os.environ, {"VLLM_HOST": host, "VLLM_API_KEY": key}), patch.object(openai, "OpenAI", capture):
                            runpy.run_path(str(path), run_name="test_module")
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0]["api_key"], key or "empty")
                        self.assertIn(expected, calls[0]["base_url"])
                        client = calls[0]["http_client"]
                        self.assertFalse(client.trust_env)
                        self.assertFalse(client.follow_redirects)
                        client.close()

    def test_generation_accepts_only_numeric_loopback_for_environment_key(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = object
        with patch.dict(sys.modules, {"transformers": transformers, "tiktoken": types.ModuleType("tiktoken")}):
            select = runpy.run_path(str(ROOT / "src/generation/run_generation.py"))["select_api_key"]
        args = types.SimpleNamespace(local_vllm=True, openai_key="fallback", openai_base_url="")
        with patch.dict(os.environ, {"VLLM_API_KEY": SYNTHETIC_KEY}):
            for url in ("http://127.0.0.1:8001/v1", "http://[::1]:8001/v1"):
                args.openai_base_url = url
                self.assertEqual(select(args), SYNTHETIC_KEY)
            for url in ("http://localhost:8001/v1", "http://example.com/v1", "http://user@127.0.0.1/v1", "http://127.0.0.1:99999/v1", "http://[::1%25lo]/v1"):
                args.openai_base_url = url
                with self.assertRaises(ValueError):
                    select(args)
            args.local_vllm = False
            self.assertEqual(select(args), "fallback")
        args.local_vllm = True
        args.openai_base_url = "http://127.0.0.1/v1"
        with patch.dict(os.environ, {"VLLM_API_KEY": ""}):
            self.assertEqual(select(args), "fallback")

    def test_evaluator_separates_hosted_and_local_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            hyp = Path(temp) / "hyp.jsonl"
            ref = Path(temp) / "ref.json"
            hyp.write_text(json.dumps({"question_id": "q", "hypothesis": "a"}) + "\n")
            ref.write_text(json.dumps([{"question_id": "q", "question_type": "single-session-user", "question": "q", "answer": "a"}]))
            for model, local in (("llama-3.1-70b-instruct", True), ("gpt-4o", False)):
                with self.subTest(local=local):
                    calls = []
                    def capture(**kwargs):
                        calls.append(kwargs)
                        message = types.SimpleNamespace(content="yes")
                        completion = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
                        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: completion)))
                    env = {"VLLM_API_KEY": SYNTHETIC_KEY, "VLLM_HOST": "::1", "OPENAI_API_KEY": "hosted-test-key"}
                    with patch.dict(os.environ, env), patch.object(openai, "OpenAI", capture), patch.object(sys, "argv", ["evaluate_qa.py", model, str(hyp), str(ref)]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        runpy.run_path(str(ROOT / "src/evaluation/evaluate_qa.py"), run_name="__main__")
                    self.assertEqual(calls[0]["api_key"], SYNTHETIC_KEY if local else "hosted-test-key")
                    if local:
                        self.assertIn("[::1]", calls[0]["base_url"])
                        self.assertFalse(calls[0]["http_client"].trust_env)
                        self.assertFalse(calls[0]["http_client"].follow_redirects)
                        calls[0]["http_client"].close()
                    else:
                        self.assertIsNone(calls[0]["base_url"])
                        self.assertNotIn("http_client", calls[0])
    def test_proxy_and_redirect_do_not_receive_local_key(self):
        with server() as (origin, origin_requests), server() as (proxy, proxy_requests):
            env = {"HTTP_PROXY": proxy, "http_proxy": proxy, "NO_PROXY": "", "no_proxy": "", "ALL_PROXY": "", "all_proxy": ""}
            with patch.dict(os.environ, env):
                # Negative control proves the proxy trap can receive a key.
                with openai.OpenAI(api_key=SYNTHETIC_KEY, base_url=origin, max_retries=0, http_client=openai.DefaultHttpxClient(trust_env=True)) as control:
                    control.get("/probe", cast_to=dict)
                with openai.OpenAI(api_key=SYNTHETIC_KEY, base_url=origin, max_retries=0, http_client=openai.DefaultHttpxClient(trust_env=False, follow_redirects=False)) as candidate:
                    candidate.get("/probe", cast_to=dict)
            self.assertEqual(proxy_requests, [True])
            self.assertEqual(origin_requests, [True])
        with server() as (attacker, attacker_requests):
            with server(response=307, location=attacker) as (redirector, redirect_requests):
                with openai.OpenAI(api_key=SYNTHETIC_KEY, base_url=redirector, max_retries=0, http_client=openai.DefaultHttpxClient(trust_env=False, follow_redirects=False)) as candidate:
                    with self.assertRaises(openai.APIStatusError) as caught:
                        candidate.get("/probe", cast_to=dict)
                    self.assertEqual(caught.exception.status_code, 307)
                self.assertEqual(redirect_requests, [True])
                self.assertEqual(attacker_requests, [])


if __name__ == "__main__":
    unittest.main()
