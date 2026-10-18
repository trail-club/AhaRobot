#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$repo_root/tools/perception/macos/setup.sh"
exec uv run --project "$perception_project" --locked --no-sync \
  python -m unittest discover \
  -s "$repo_root/tests/perception" -p 'test_*.py' -v
