#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$repo_root/tools/perception/macos/setup.sh"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$repo_root/tools/perception/macos/sam31/interactive.py" "$@"
