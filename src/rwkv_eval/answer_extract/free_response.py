"""Free-response normalization used by RWKV generative evaluations."""

from __future__ import annotations

import re


def extract_free_response(text: str, *, require_think_close: bool = False) -> str:
    """Remove reasoning wrapper and conversation spillover from a completion."""
    if require_think_close and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.split("\nUser:", 1)[0]
    text = text.strip()
    if text.startswith(">"):
        text = text[1:].lstrip()
    return text


def extract_math_answer(text: str) -> str:
    """Return the answer portion while preserving expressions for math metrics."""
    text = extract_free_response(text)
    boxed = list(re.finditer(r"\\boxed\s*\{(.+?)\}", text, re.S))
    return boxed[-1].group(1).strip() if boxed else text


def verify_math_answer(gold: str, text: str) -> bool:
    """Verify a MATH500-style response using math_verify when available."""
    try:
        from math_verify import parse, verify

        expected = parse(f"$\\boxed{{{gold}}}$")
        predicted = parse(extract_free_response(text))
        return bool(predicted and verify(expected, predicted, strict=False))
    except Exception:
        return False


def extract_answer(text: str) -> str:
    return extract_free_response(text)
