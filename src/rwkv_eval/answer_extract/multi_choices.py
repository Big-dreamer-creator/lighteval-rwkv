"""Robust multiple-choice extraction based on Albatross' GPQA extractor."""

from __future__ import annotations

import re


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MARKUP = re.compile(r"\*\*|__|`")
_PRIMARY = (
    re.compile(r"\\boxed\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?\(?\s*([A-Z])\s*\)?\s*\}?\s*\}", re.I),
    re.compile(r"(?:final\s+answer|correct\s+answer|answer)\s*(?:(?:choice|option)\s*)?(?:is\s*|[:=]\s*)(?:(?:choice|option)\s*)?\(?\s*([A-Z])", re.I),
    re.compile(r"(?:(?:choice|option)\s*)?\(?\s*([A-Z])\s*\)?\s+is\s+(?:the\s+)?(?:final\s+|correct\s+)?answer", re.I),
)
_FALLBACK = (
    re.compile(r"\b(?:choose|select|pick)\s+(?:(?:choice|option|answer)\s*)?[:=]?\s*\(?\s*([A-Z])", re.I),
    re.compile(r"\b(?:corresponds?|maps?)\s+to\s+(?:(?:choice|option|answer)\s*)?\(?\s*([A-Z])", re.I),
    re.compile(r"\b(?:therefore|thus|hence|so|consequently)[,:]?\s+(?:the\s+)?(?:correct\s+|final\s+)?(?:answer|choice|option)\s+(?:is|would\s+be)\s+\(?\s*([A-Z])", re.I),
)
_THINK_FINAL = re.compile(
    r"(?i:\b(?:final\s+answer|correct\s+(?:answer|choice|option))\s*"
    r"(?:is\s*|[:=]\s*)?(?:\\boxed\s*\{\s*)?(?:\\(?:text|mathrm)\s*\{\s*)?\(?\s*([A-Z]))"
)


def _last_match(text: str, patterns: tuple[re.Pattern[str], ...], letters: str) -> str | None:
    matches = [
        (match.start(), match.group(1).upper())
        for pattern in patterns
        for match in pattern.finditer(text)
        if match.group(1).upper() in letters
    ]
    return max(matches, key=lambda item: item[0])[1] if matches else None


def extract_choice_answer(text: str, choice_count: int, *, require_think_close: bool = False) -> str | None:
    """Return the final A/B/C... answer, with Albatross-style fallbacks."""
    letters = LETTERS[:choice_count]
    if not letters or (require_think_close and "</think>" not in text):
        return None

    before_think, think_closed, answer_text = text.rpartition("</think>")
    answer_text = _MARKUP.sub("", answer_text if think_closed else text)
    answer = _last_match(answer_text, _PRIMARY, letters) or _last_match(answer_text, _FALLBACK, letters)
    if answer:
        return answer

    for line in reversed(answer_text.splitlines()):
        match = re.fullmatch(r"\s*(?:final\s+answer\s*[:=]?\s*)?[\[(]?([A-Z])[\])]?[.!]?\s*", line, re.I)
        if match and match.group(1).upper() in letters:
            return match.group(1).upper()

    if think_closed:
        matches = [match.group(1).upper() for match in _THINK_FINAL.finditer(_MARKUP.sub("", before_think))]
        if matches and matches[-1] in letters:
            return matches[-1]
    return None


def extract_choice_indices(text: str, choice_count: int, *, require_think_close: bool = False) -> tuple[int, ...] | None:
    answer = extract_choice_answer(text, choice_count, require_think_close=require_think_close)
    return None if answer is None else (LETTERS.index(answer),)


def extract_answer(text: str, choices: list[str] | tuple[str, ...]) -> str | None:
    """Compatibility helper returning the selected choice text."""
    indices = extract_choice_indices(text, len(choices))
    if not indices:
        return None
    return str(choices[indices[0]])
