#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
requirements="$repo_root/tools/perception/macos/requirements.txt"
if [[ -n "${PERCEPTION_PYTHON:-}" ]]; then
  python_bin="$PERCEPTION_PYTHON"
  managed_venv=false
else
  venv="$repo_root/.venv-perception-macos"
  python_bin="$venv/bin/python"
  managed_venv=true
  if [[ ! -x "$python_bin" ]]; then
    echo '[perception] Creating Python 3.11 environment...'
    base_python=""
    for candidate in "${PERCEPTION_BASE_PYTHON:-python3.11}" /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
      if "$candidate" -c 'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
        base_python="$candidate"
        break
      fi
    done
    if [[ -z "$base_python" ]]; then
      echo '[perception] Python 3.11 is required to create the environment.' >&2
      exit 1
    fi
    # A separately built RealSense binding may already be in this Python's
    # site-packages. Keep it visible while installing NumPy in the venv.
    "$base_python" -m venv --system-site-packages "$venv"
  fi
fi

if ! "$python_bin" -c 'import numpy, websocket' >/dev/null 2>&1; then
  if [[ "$managed_venv" == false ]]; then
    echo "[perception] Install dependencies in PERCEPTION_PYTHON: $python_bin -m pip install -r $requirements" >&2
    exit 1
  fi
  echo '[perception] Installing Python dependencies...'
  "$python_bin" -m pip --version >/dev/null 2>&1 || "$python_bin" -m ensurepip --upgrade
  "$python_bin" -m pip install -r "$requirements"
fi

exec "$python_bin" "$repo_root/tools/perception/macos/start.py" "$@"
