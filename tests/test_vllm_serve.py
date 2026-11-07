import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ("serve_vllm.sh", ["0", "llama-3-8b-instruct"]),
    ("serve_vllm_with_maxlen.sh", ["0", "llama-3-8b-instruct", "2048"]),
)


class VllmServeTest(unittest.TestCase):
    def run_script(self, name, args, host=None, api_key=None):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            shim = tmp / "python"
            shim.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
            shim.chmod(0o755)
            env = {"PATH": str(tmp) + os.pathsep + os.defpath, "CAPTURE": str(tmp / "args")}
            if host is not None:
                env["VLLM_HOST"] = host
            if api_key is not None:
                env["VLLM_API_KEY"] = api_key
            result = subprocess.run(
                ["bash", str(ROOT / "src" / "utils" / name), *args],
                cwd=ROOT / "src" / "utils", env=env, text=True, capture_output=True,
            )
            args_file = tmp / "args"
            return result, args_file.read_text().splitlines() if args_file.exists() else []

    def test_local_by_default_and_explicit_remote_requires_key(self):
        for name, args in SCRIPTS:
            with self.subTest(script=name):
                result, argv = self.run_script(name, args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
                result, argv = self.run_script(name, args, host="0.0.0.0")
                self.assertEqual(result.returncode, 2)
                self.assertEqual(argv, [])
                result, argv = self.run_script(name, args, host="0.0.0.0", api_key="synthetic")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")
                self.assertNotIn("synthetic", argv)


if __name__ == "__main__":
    unittest.main()
