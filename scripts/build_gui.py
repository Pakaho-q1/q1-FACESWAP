"""
build_gui.py
============
Build the Tauri desktop GUI for q1-FaceSwap into a standalone binary.

Requires: Node.js + npm + Rust toolchain installed on the system.
Output: webui/dist/gui.exe (copied from target/release/)

Usage:
    python scripts/build_gui.py
    python scripts/build_gui.py --skip-npm-install
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd, cwd, label):
    print(f"[{label}] Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=cwd, shell=(sys.platform == "win32"))
        print(f"[{label}] Done.")
    except subprocess.CalledProcessError as e:
        print(f"[{label}] FAILED (exit code {e.returncode})")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Build q1-FaceSwap Tauri desktop GUI")
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Skip npm install step (use if node_modules already present)",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    webui_dir = os.path.join(repo_root, "webui")

    if not os.path.isdir(webui_dir):
        print(f"[FAIL] webui directory not found at: {webui_dir}")
        sys.exit(1)

    print("Building q1-FaceSwap Desktop GUI")
    print(f"  webui dir: {webui_dir}")
    print()

    if not args.skip_npm_install:
        run(["npm", "install"], cwd=webui_dir, label="npm install")

    run(["npm", "run", "tauri", "build"], cwd=webui_dir, label="tauri build")

    # Locate the built binary
    release_dir = os.path.join(webui_dir, "src-tauri", "target", "release")
    dest_dir = os.path.join(webui_dir, "dist")
    os.makedirs(dest_dir, exist_ok=True)
    src_exe = os.path.join(release_dir, "app.exe")
    dest_exe = os.path.join(dest_dir, "gui.exe")
    if os.path.isfile(src_exe):
        import shutil
        shutil.copy2(src_exe, dest_exe)
        print(f"[SUCCESS] GUI binary copied to: {dest_exe}")
        print(f"Run it: python faceswap.py gui")
    else:
        print(f"[WARN] app.exe not found in {release_dir}")
        print("Check the Tauri build output above for errors.")


if __name__ == "__main__":
    main()
