#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
project="$repo_root/tools/perception/macos/audio"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$repo_root/.venv-audio-macos}"
exec uv run --project "$project" --locked python "$project/audio_ui.py" "$@"
