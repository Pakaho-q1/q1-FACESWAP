import os
import subprocess
import sys
import tempfile
import unittest


class ModuleEntrypointTests(unittest.TestCase):
    def test_python_m_core_dry_run(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        with tempfile.TemporaryDirectory() as temp_input, tempfile.TemporaryDirectory() as temp_output:
            cmd = [
                sys.executable,
                "-m",
                "core",
                "--face-model-name",
                "dryrun",
                "--format",
                "1",
                "--input-path",
                temp_input,
                "--output-path",
                temp_output,
                "--use-swaper",
                "false",
                "--use-restore",
                "false",
                "--use-parser",
                "false",
                "--dry-run",
                "true",
            ]
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("DRY_RUN", proc.stdout)


if __name__ == "__main__":
    unittest.main()
