#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPECTED_COUNT=3172
exec "${SCRIPT_DIR}/11_run_dspy_react_agent_daily_o3_gemini25flash_basic_tools_free_react_6turn.sh" "$@"
