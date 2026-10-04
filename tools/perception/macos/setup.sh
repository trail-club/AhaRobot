#!/usr/bin/env bash
# Also sourced by start.sh and test.sh to select the same uv environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
perception_project="$repo_root/tools/perception/macos"
perception_python="${PERCEPTION_PYTHON:-${PERCEPTION_BASE_PYTHON:-3.11}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$repo_root/.venv-perception-macos}"

if ! command -v uv >/dev/null 2>&1; then
  echo '[perception] uv is required; install it before running setup.sh.' >&2
  return 1 2>/dev/null || exit 1
fi

# Keep the independently built RealSense SDK/binding in an existing environment.
uv sync --project "$perception_project" --locked --inexact --python "$perception_python"
