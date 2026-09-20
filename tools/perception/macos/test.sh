#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -n "${PERCEPTION_PYTHON:-}" ]]; then
  python_bin="$PERCEPTION_PYTHON"
elif [[ -x "$repo_root/.venv-perception-macos/bin/python" ]]; then
  python_bin="$repo_root/.venv-perception-macos/bin/python"
else
  python_bin=python3.11
fi

"$python_bin" -m unittest discover \
  -s "$repo_root/tests/perception" -p 'test_*.py' -v
