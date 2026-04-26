import os
import subprocess
import sys
import tempfile
import unittest


class DryRunConfigValidationTests(unittest.TestCase):
    def test_dry_run_exits_successfully(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(repo_root, "face_swap_unified.py")

        with tempfile.TemporaryDirectory() as temp_input, tempfile.TemporaryDirectory() as temp_output:
            cmd = [
                sys.executable,
                script_path,
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

    def test_print_effective_config(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(repo_root, "face_swap_unified.py")

        with tempfile.TemporaryDirectory() as temp_input, tempfile.TemporaryDirectory() as temp_output:
            cmd = [
                sys.executable,
                script_path,
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
                "--high-watermark",
                "12",
                "--low-watermark",
                "4",
                "--switch-cooldown-s",
                "0.35",
                "--print-effective-config",
                "true",
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
            self.assertIn("EFFECTIVE_CONFIG", proc.stdout)
            self.assertIn("HIGH_WATERMARK", proc.stdout)
            self.assertIn("LOW_WATERMARK", proc.stdout)
            self.assertIn("SWITCH_COOLDOWN_S", proc.stdout)


if __name__ == "__main__":
    unittest.main()
