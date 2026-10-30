# Efficient Agentic Audio-Visual Question Answering

Code for the paper submission _Efficient Agentic Audio-Visual Question
Answering through Toolset Pruning and Modality Consolidation_.

## Environment setup

The experiments use Python 3.11 and the following principal packages:

| Package | Version | Role |
| --- | ---: | --- |
| DSPy | 3.2.1 | Compact agent |
| LiteLLM | 1.82.6 | Model and API routing |
| LangChain | 1.1.0 | OmniAgent reproduction |

Create a Python 3.11 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the media tools on Ubuntu/Debian and check that both executables are available:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
ffprobe -version
```

Additional dependencies for each inference pipeline will be added with its
source code. The experiments use o3 as the text-only planner and
Gemini-2.5-Flash as the audio-visual perception
model. The compact agent uses the official OpenAI API for o3 and the official
Vertex AI Gemini `generateContent` API. Configure them in a local `.env` file:

```bash
OPENAI_API_KEY=...
GEMINI_BASE_URL=https://aiplatform.googleapis.com/v1/projects/<project-id>/locations/<location>/publishers/google
GEMINI_API_KEY=...
GEMINI_MEDIA_MODE=inline
```

Do not commit `.env` or other credential files.

## Main results

Accuracy and inference efficiency across three audio-visual QA benchmarks are
reported below. All direct Gemini-2.5-Flash and OmniAgent results are obtained
from our own runs; OmniAgent is reproduced based on its released code. All
efficiency values are averages per question. Input and output tokens include
all model calls. Latency is the mean per-question retry-adjusted wall-clock
time (serialized execution time per question). Bold indicates the highest accuracy within each benchmark; underlining
marks the more efficient agentic system.

| Benchmark | Method | Inference paradigm | # Tools | Acc. (%) ↑ | Calls ↓ | Input / output tok. ↓ | Est. cost ($) ↓ | Latency (s) ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Daily-Omni (n=1,197) | Gemini-2.5-Flash | Direct OmniLLM | — | 77.03 | 0.000 | 5.048k / 0.006k | 0.00154 | — |
|  | OmniAgent | Specialized agent | 8 | 77.53 | 7.270 | 49.661k / 7.462k | 0.07142 | 118.32 |
|  | Ours | Compact agent | <u>2</u> | **78.86** | <u>1.550</u> | <u>13.000k / 1.670k</u> | <u>0.01910</u> | <u>14.90</u> |
| AVUT (n=1,734) | Gemini-2.5-Flash | Direct OmniLLM | — | 79.35 | 0.000 | 11.457k / 0.149k | 0.00437 | — |
|  | OmniAgent | Specialized agent | 8 | 77.16 | 6.382 | 33.898k / 4.272k | 0.08312 | 107.76 |
|  | Ours | Compact agent | <u>2</u> | **80.57** | <u>1.777</u> | <u>16.814k / 1.432k</u> | <u>0.02047</u> | <u>27.55</u> |
| WorldSense (n=3,172) | Gemini-2.5-Flash | Direct OmniLLM | — | 56.43 | 0.000 | 22.517k / 0.303k | 0.00874 | — |
|  | OmniAgent | Specialized agent | 8 | **59.30** | 7.358 | 69.986k / 5.558k | 0.11414 | 143.64 |
|  | Ours | Compact agent | <u>2</u> | 58.39 | <u>1.986</u> | <u>39.017k / 2.019k</u> | <u>0.03139</u> | <u>31.29</u> |

## Compact agent inference

The compact agent uses an o3 planner with medium reasoning effort and Gemini
2.5 Flash for audio-visual perception. Its only tools are `ask_caption` and
`ask_perception`. The first caption call is read from a question-scoped cache;
later caption or perception calls use Gemini. Each question has a budget of six
tool calls, followed by a forced final-answer step when needed.

The input is a cut-level JSONL file. Each row must have an `id` and one
supervision. The question is read from `supervisions[0].text` (or its
`custom.question` fallback); answer choices, gold answer, and local
`video_path` and `audio_path` values live under `supervisions[0].custom`.
Media paths may differ from the paths used in our experiments.

The caption cache directory must contain `manifest.json` and one
`<question_id>.json` file for every input row. The runner validates the cache
schema, question coverage, caption prompt, and content digest before inference.
The cached caption prompt must match the released prompt exactly.

Set credentials in `.env`:

```bash
OPENAI_API_KEY=...
GEMINI_BASE_URL=https://aiplatform.googleapis.com/v1/projects/<project-id>/locations/<location>/publishers/google
GEMINI_API_KEY=...
GEMINI_MEDIA_MODE=inline
```

For Vertex authentication, `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json`
may be used instead of `GEMINI_API_KEY`. Local media is embedded as native
`inlineData` by default. To use Vertex `fileData`, set `GEMINI_MEDIA_MODE=gcs`
and provide matching `GEMINI_LOCAL_DATA_ROOT` and `GEMINI_GCS_DATA_ROOT` values.

For each benchmark, set its input, caption cache, and a writable output
directory, then run the corresponding script. For example:

```bash
export INPUT_JSONL=/path/to/daily_omni_test.jsonl
export CAPTION_CACHE_DIR=/path/to/daily_omni_caption_cache
export OUTPUT_DIR=outputs/daily_omni
bash scripts/11_run_dspy_react_agent_daily_o3_gemini25flash_basic_tools_free_react_6turn.sh
```

```bash
export INPUT_JSONL=/path/to/avut_test.jsonl
export CAPTION_CACHE_DIR=/path/to/avut_caption_cache
export OUTPUT_DIR=outputs/avut
bash scripts/11_run_dspy_react_agent_avut_o3_gemini25flash_basic_tools_free_react_6turn.sh
```

```bash
export INPUT_JSONL=/path/to/worldsense_test.jsonl
export CAPTION_CACHE_DIR=/path/to/worldsense_caption_cache
export OUTPUT_DIR=outputs/worldsense
bash scripts/11_run_dspy_react_agent_worldsense_o3_gemini25flash_basic_tools_free_react_6turn.sh
```

Each benchmark script checks the Table 1 test-set size before running: 1,197
for Daily-Omni, 1,734 for AVUT, and 3,172 for WorldSense. Set
`INFERENCE_NUM_THREADS` to control concurrent questions; it defaults to four.
Existing per-question files in `OUTPUT_DIR` are used as resumable checkpoints.
The aggregate result is written to `OUTPUT_DIR/output_test.jsonl`. The script
returns a failure status if that file contains an error, empty response, or
unparseable final answer; rerunning it retries failed checkpoints.

To validate paths and print the resolved command without making model requests,
prefix any command with `DRY_RUN=true`. To run selected questions from a full
benchmark input, append `--sample-ids ID1 ID2` to the command. A dry run only
checks configuration; it is not an inference result.

