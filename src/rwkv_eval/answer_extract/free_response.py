"""Answer extraction for RWKV reasoning and short-answer evaluations."""

from __future__ import annotations

import re


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_BOXED = re.compile(r"\\boxed\s*\{")
_FINAL = re.compile(
    r"(?im)^\s*(?:final\s+answer|the\s+answer|answer|final|therefore|thus|so)"
    r"\s*(?:is\s*)?(?::|=)?\s*(.+?)\s*$"
)
_LATEX_EXPRESSION = re.compile(r"\\(?:sqrt|frac|dfrac|tfrac)\s*(?:\{[^{}]*\}){1,2}")
_SHORT_TEXT = re.compile(r"[A-Za-z][A-Za-z' .-]{0,79}")
_MATH_TOKEN = re.compile(
    r"\\(?:frac|dfrac|tfrac)\s*\{[^{}]*\}\s*\{[^{}]*\}"
    r"|[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)(?:\s*%|\s*/\s*\d+)?"
)
_MATH_FUNCTION = re.compile(r"(?<!\\)\b(sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|log|ln|exp)\b(?=\s*[A-Za-z(^\\{])")


def extract_free_response(text: str, *, require_think_close: bool = False) -> str:
    """Remove RWKV reasoning tags and assistant-message spillover."""
    if require_think_close and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think" in text:
        # An unterminated reasoning trace reached the output limit.  It is a
        # failed answer, not a valid fallback response.
        return ""
    if "\nUser:" in text:
        text = text.split("\nUser:", 1)[0]
    if "\nAssistant:" in text:
        text = text.rsplit("\nAssistant:", 1)[1]
    text = text.strip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:") :].lstrip()
    if text.startswith(">"):
        text = text[1:].lstrip()
    return text


def _boxed_content(text: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` expression."""
    valid: list[tuple[int, int, str]] = []
    for match in _BOXED.finditer(text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    valid.append((index, match.start(), text[start:index].strip()))
                    break
    if not valid:
        return None
    last_end = max(item[0] for item in valid)
    if any(match.start() > last_end for match in _BOXED.finditer(text)):
        return None
    return min((item for item in valid if item[0] == last_end), key=lambda item: item[1])[2]


def extract_math_answer(text: str, *, allow_text: bool = False) -> str | None:
    """Extract a concise math answer, or return ``None`` on extraction failure.

    The raw completion and the concise answer are intentionally separate.  A
    failed extraction must not fall back to uploading the whole final line.
    """
    text = extract_free_response(text)
    if not text:
        return None
    boxed = _boxed_content(text)
    if boxed:
        boxed = boxed.replace("，", ",").strip("$ ")
        return (
            re.sub(r"^\\(?:text|mathrm)\{(.*)\}$", r"\1", boxed) if allow_text else _MATH_FUNCTION.sub(r"\\\1", boxed)
        )
    if _BOXED.search(text):
        return None
    marked = _FINAL.findall(text)
    if marked:
        text = marked[-1].strip().strip("`$ ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_line = lines[-1].replace("$", "").replace("，", ",").strip("` ") if lines else ""
    # ASDiv's native answer field also includes names, colors and Yes/No.
    # Accept concise textual answers there, not a worked-solution fallback.
    if allow_text and _SHORT_TEXT.fullmatch(final_line) and len(final_line.split()) <= 5:
        return final_line.rstrip(" .") or None
    # Explanatory clauses may contain numbers unrelated to the answer, e.g.
    # "The remainder is 0, because 54 is divisible by 6." They are not a
    # mathematical-expression fallback. Preserve the assertion before them.
    final_line = re.split(r",?\s+\b(?:because|since)\b", final_line, maxsplit=1, flags=re.IGNORECASE)[0]
    final_line = _MATH_FUNCTION.sub(r"\\\1", final_line)
    # Preserve a complete concise expression (including nested LaTeX) before
    # looking for an individual token.  This also makes gold processing
    # idempotent when a worked solution has already been reduced to its box.
    expression = re.sub(r"\\(?:text|mathrm|operatorname)\{[^{}]*\}", "", final_line)
    expression = re.sub(r"\\[A-Za-z]+|\b(?:sin|cos|tan|log|ln|exp)\b", "", expression)
    if (
        not re.search(r"[A-Za-z]{3,}", expression)
        and re.fullmatch(r"[A-Za-z0-9\\\s^_{}()\[\]<>|+=*/.,:%-]+", final_line)
        and re.search(r"[A-Za-z0-9\\]", final_line)
        and (re.search(r"[=^+*/\\,:|\-]", final_line) or re.fullmatch(r"[A-Za-z]", final_line))
    ):
        return final_line.rstrip(" .") or None
    latex = _LATEX_EXPRESSION.search(final_line)
    if latex:
        return latex.group(0).strip()
    candidates = _MATH_TOKEN.findall(final_line or text)
    return candidates[-1].strip().replace(",", "") if candidates else None


def _ordered_named_answers(text: str, rendered_prompt: str) -> str | None:
    """Reduce named ASDiv values in the order requested by its question.

    Prototype: balanced-box extraction followed by concise math extraction.
    Only actual prompt evidence supplies the ordering; no gold values are used.
    """
    parts = text.replace(r"\quad", "").replace(r"\qquad", "").split(",")
    if len(parts) < 2:
        return None
    question = rendered_prompt.rsplit("Question:", 1)[-1].split("\nAnswer:", 1)[0].casefold()
    ordered = []
    for part in parts:
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z' ]*?)\s*[:=]\s*(.+?)\s*", part)
        if match is None:
            return None
        position = re.search(r"\b" + re.escape(match[1].strip().casefold()) + r"\b", question)
        value = extract_math_answer(match[2])
        if position is None or value is None:
            return None
        ordered.append((position.start(), value))
    return "; ".join(value for _, value in sorted(ordered))


def answer_extractor_for_benchmark(selector: str) -> str | None:
    """Select the RWKV output contract, independently of request routing."""
    name = selector.split("|", 1)[0].split(":", 1)[0].casefold()
    if name in {
        "aimo_progress_prize_1",
        "aime24",
        "aime25",
        "gsm8k",
        "gsm_plus",
        "arithmetic",
        "math",
        "math_500",
        "olympiad_bench",
    }:
        return "math"
    if name == "asdiv":
        return "short_answer"
    if name == "gpqa":
        return "choice"
    if name == "lcb":
        return "code"
    return None


def extract_answer(
    text: str, extractor: str | None, choice_count: int = 0, *, rendered_prompt: str | None = None
) -> str | None:
    """Shared RWKV metric/detail extraction; failure stays ``None``."""
    rendered_prompt = rendered_prompt or ""
    # Fake Think intentionally leaves the final '>' for generation. Parse the
    # observed prompt/output boundary without changing either uploaded string.
    if rendered_prompt.endswith("</think") and text.startswith(">"):
        rendered_prompt += text[0]
    if text.rfind("<think") > text.rfind("</think>"):
        return None
    # Open Think may start in the prompt, whereas Fake Think can close there.
    # Use captured template evidence, not a mode label, to distinguish them.
    require_think_close = (
        "</think>" not in rendered_prompt
        or rendered_prompt.rfind("<think") > rendered_prompt.rfind("</think>")
        or "<think" in text
    )
    if extractor == "choice":
        from src.rwkv_eval.answer_extract.multi_choices import extract_choice_answer

        return extract_choice_answer(text, choice_count, require_think_close=require_think_close)
    text = extract_free_response(text, require_think_close=require_think_close)
    if extractor in {"math", "short_answer"}:
        answer = extract_math_answer(text, allow_text=extractor == "short_answer")
        if extractor == "short_answer" and answer is not None:
            return _ordered_named_answers(answer, rendered_prompt) or answer
        return answer
    if extractor == "code":
        return extract_code_answer(text) or None
    return text or None


def verify_math_answer(gold: str, text: str, *, allow_text: bool = False) -> bool:
    """Albatross-style exact agreement followed by native math equivalence.

    Prototype: Metrics.pass_at_k_math's MultilingualExtractiveMatchMetric.
    Callers supply concise values; box them for the native LaTeX extractor.
    """
    if not gold or not text:
        return False
    if gold.strip() == text.strip():
        return True
    if allow_text and ";" in gold:
        golds = gold.split(";")
        answer = text.strip("()[] ")
        answers = answer.split(";") if ";" in answer else answer.split(",")
        return len(golds) == len(answers) and all(
            verify_math_answer(g, a, allow_text=True) for g, a in zip(golds, answers)
        )
    if allow_text and _SHORT_TEXT.fullmatch(gold.strip()):
        # Follow the native exact-match contract for textual targets; names
        # must not become products of symbols in the mathematical comparator.
        return False
    gold = _MATH_FUNCTION.sub(r"\\\1", gold)
    text = _MATH_FUNCTION.sub(r"\\\1", text)
    from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
    from lighteval.metrics.utils.extractive_match_utils import LatexExtractionConfig
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import Doc
    from lighteval.utils.language import Language

    comparator = MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        gold_extraction_target=[LatexExtractionConfig()],
        pred_extraction_target=[LatexExtractionConfig()],
        precision=6,
    )
    doc = Doc(query="", choices=[f"$\\boxed{{{gold}}}$"], gold_index=0)
    response = ModelResponse(text=[f"$\\boxed{{{text}}}$"])
    return comparator.compute(doc, response) == 1.0


def extract_code_answer(text: str) -> str:
    """Extract the final fenced program used by LiveCodeBench."""
    text = extract_free_response(text)
    blocks = _CODE_BLOCK.findall(text)
    return blocks[-1].strip() if blocks else ""
