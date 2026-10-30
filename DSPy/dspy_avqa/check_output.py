"""Check that a compact-agent output contains only completed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .program import normalize_option_letter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()

    rows = []
    with args.output_jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"Inference output is empty: {args.output_jsonl}")

    failures: list[str] = []
    for row in rows:
        question_id = str(row.get("video_id") or "<unknown>")
        response = str((row.get("question_data") or {}).get("response") or "").strip()
        if not response or response.startswith("[ERROR]"):
            failures.append(question_id)
            continue
        if not normalize_option_letter(response):
            failures.append(question_id)

    if failures:
        preview = ", ".join(failures[:10])
        raise SystemExit(
            f"Inference produced {len(failures)} failed or unparseable row(s): {preview}"
        )
    print(f"Validated inference output: {len(rows)} completed row(s)")


if __name__ == "__main__":
    main()
