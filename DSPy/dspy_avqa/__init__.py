"""Compact-agent inference package for audio-visual question answering."""

from .program import AVQADSPyReActProgram, normalize_option_letter
from .runner import parse_args, run_batch, run_one

__all__ = [
    "AVQADSPyReActProgram",
    "normalize_option_letter",
    "parse_args",
    "run_batch",
    "run_one",
]
