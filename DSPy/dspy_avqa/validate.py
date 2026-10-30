"""Validate compact-agent inputs without making model requests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .caption_cache import (
    effective_caption_prompt_from_yaml,
    validate_caption_cache_coverage,
)
from .data import build_input_state, read_jsonl


def validate_inputs(
    input_jsonl: Path,
    caption_cache_dir: Path,
    prompt_yaml: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Validate input rows, local video paths, and the caption cache."""
    cuts = read_jsonl(input_jsonl)
    if len(cuts) != expected_count:
        raise ValueError(
            f"Input row count mismatch: expected {expected_count}, found {len(cuts)}"
        )

    question_ids: list[str] = []
    for cut in cuts:
        payload = build_input_state(cut, audio_caption_dir=None)
        question_ids.append(payload["question_id"])
        video_path = Path(payload["video_path"]).expanduser()
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video for question {payload['question_id']} not found: {video_path}"
            )

    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Input JSONL contains duplicate question IDs")

    coverage = validate_caption_cache_coverage(
        caption_cache_dir,
        question_ids,
        expected_prompt=effective_caption_prompt_from_yaml(prompt_yaml),
    )
    return {
        "input_rows": len(cuts),
        "unique_question_ids": len(set(question_ids)),
        "caption_cache_validated": coverage["validated"],
        "caption_cache_manifest_sha256": coverage["manifest_sha256"],
        "caption_cache_content_sha256": coverage["content_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--caption-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    result = validate_inputs(
        args.input_jsonl,
        args.caption_cache_dir,
        args.prompt_yaml,
        args.expected_count,
    )
    print(
        "Validated compact-agent inputs: "
        f"rows={result['input_rows']}, "
        f"question_ids={result['unique_question_ids']}, "
        f"caption_cache={result['caption_cache_validated']}"
    )
    print(f"Caption cache manifest SHA-256: {result['caption_cache_manifest_sha256']}")
    print(f"Caption cache content SHA-256: {result['caption_cache_content_sha256']}")


if __name__ == "__main__":
    main()
