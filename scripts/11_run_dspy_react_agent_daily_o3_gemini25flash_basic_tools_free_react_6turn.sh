#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the o3 planner.}"
: "${GEMINI_BASE_URL:?Set GEMINI_BASE_URL to the official Vertex AI publisher endpoint.}"
if [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
  echo "Set either GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS for official Vertex Gemini." >&2
  exit 1
fi
: "${INPUT_JSONL:?Set INPUT_JSONL to the benchmark cut JSONL.}"
: "${CAPTION_CACHE_DIR:?Set CAPTION_CACHE_DIR to the question-scoped caption cache.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints, logs, and output_test.jsonl.}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1197}"

export PLANNER_API_KEY="${OPENAI_API_KEY}"
export PLANNER_PROVIDER="openai"
unset OPENAI_API_BASE OPENAI_BASE_URL PLANNER_API_BASE
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="official"
export GEMINI_MEDIA_MODE="${GEMINI_MEDIA_MODE:-inline}"
export GEMINI_MODEL="gemini-2.5-flash"
export CAPTIONER_GEMINI_MODEL="gemini-2.5-flash"
export GEMINI_RESPONSE_ERROR_SENSITIVE="${GEMINI_RESPONSE_ERROR_SENSITIVE:-false}"

PLANNER_CONFIG_YAML="${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_openai_o3_medium.yaml"
GEMINI_CONFIG_YAML="${REPO_DIR}/DSPy/dspy_avqa/yamls/config_gemini25_flash_env.yaml"
PROMPT_YAML="${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_caption_in_task.yaml"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/output_test.jsonl}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/compact_agent.log}"
INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${INFERENCE_NUM_THREADS}}"

actual_count="$(awk 'NF {count++} END {print count+0}' "${INPUT_JSONL}")"
if [ "${actual_count}" -ne "${EXPECTED_COUNT}" ]; then
  echo "Input row count mismatch: expected ${EXPECTED_COUNT}, found ${actual_count} in ${INPUT_JSONL}" >&2
  exit 2
fi
if [ "${INFERENCE_NUM_THREADS}" -lt 1 ] || [ "${INFERENCE_BATCH_SIZE}" -lt 1 ]; then
  echo "INFERENCE_NUM_THREADS and INFERENCE_BATCH_SIZE must be >= 1." >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" "${REPO_DIR}/DSPy/avqa_dspy_impl.py"
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-turns 6
  --inference-num-threads "${INFERENCE_NUM_THREADS}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --planner-config-yaml "${PLANNER_CONFIG_YAML}"
  --perception-model gemini
  --perception-config-yaml "${GEMINI_CONFIG_YAML}"
  --captioner-config-yaml "${GEMINI_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools ask_caption,ask_perception
  --caption-cache-dir "${CAPTION_CACHE_DIR}"
  --caption-cache-scope first_call_only
  --ignore-audio-caption-dir
  --gemini-api-backend official
  --gemini-media-mode "${GEMINI_MEDIA_MODE}"
  --signature-in-system-prompt
  --caption-placement task
  --print-config
)

if [ "${DRY_RUN:-false}" = "true" ]; then
  echo "Dry run: no model request will be sent."
  PYTHONPATH="${REPO_DIR}/DSPy${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
    -m dspy_avqa.validate \
    --input-jsonl "${INPUT_JSONL}" \
    --caption-cache-dir "${CAPTION_CACHE_DIR}" \
    --prompt-yaml "${PROMPT_YAML}" \
    --expected-count "${EXPECTED_COUNT}"
  print_run_info
  printf 'Command:'
  printf ' %q' "${cmd[@]}" "$@"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
{
  echo "Started: $(date -Is)"
  print_run_info
  "${cmd[@]}" "$@"
  PYTHONPATH="${REPO_DIR}/DSPy${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
    -m dspy_avqa.check_output "${OUTPUT_JSONL}"
  echo "Finished: $(date -Is)"
} 2>&1 | tee -a "${LOG_FILE}"
