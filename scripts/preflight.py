#!/usr/bin/env python3
"""Preflight check: validates Python, compileall, and requirements before deploy/restart."""
import subprocess
import sys
import os

def main():
    # 1. Print python version and cwd
    print(f"Python: {sys.version}")
    cwd = os.getcwd()
    print(f"CWD: {cwd}")

    # 2. Run compileall - exit non-zero if fails
    print("Running compileall...")
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "."],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("compileall FAILED:", result.stderr or result.stdout or "unknown")
        sys.exit(1)
    print("compileall OK")

    # 3. Validate requirements.txt parseable by pip (dry-run)
    req_path = os.path.join(cwd, "requirements.txt")
    if not os.path.exists(req_path):
        print(f"ERROR: requirements.txt not found at {req_path}")
        sys.exit(1)

    print("Validating requirements.txt (pip dry-run)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_path, "--dry-run"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            # Fallback: pip download validates requirements without installing
            print("pip dry-run failed, trying pip download...")
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                result2 = subprocess.run(
                    [sys.executable, "-m", "pip", "download", "-r", req_path, "-d", tmp],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if result2.returncode != 0:
                    print("pip download FAILED:", result2.stderr or result2.stdout or "unknown")
                    sys.exit(1)
    except subprocess.TimeoutExpired:
        print("pip validation TIMEOUT")
        sys.exit(1)
    print("requirements.txt OK")

    print("PREFLIGHT OK")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"PREFLIGHT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
