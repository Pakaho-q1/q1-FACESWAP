import os
import subprocess
import sys
import unittest


class CLISubcommandTests(unittest.TestCase):
    def test_version_subcommand(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [
            sys.executable,
            "-m",
            "core",
            "version"
        ]
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("q1-FaceSwap version", proc.stdout)

    def test_version_json_subcommand(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [
            sys.executable,
            "-m",
            "core",
            "version",
            "--json"
        ]
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn('"version":', proc.stdout)


if __name__ == "__main__":
    unittest.main()
