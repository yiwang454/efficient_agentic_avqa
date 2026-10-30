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
model. Configure the provider credentials in a local `.env` file:

```bash
ELM_API_KEY=...
GEMINI_API_KEY=...
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

The compact-agent inference scripts for Daily-Omni, AVUT, and WorldSense will
be linked here when they are added in the next release stage.

