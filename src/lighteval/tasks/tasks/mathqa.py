"""
name:
Mathqa

dataset:
allenai/math_qa

abstract:
large-scale dataset of math word problems.  Our dataset is gathered by using a
new representation language to annotate over the AQuA-RAT dataset with
fully-specified operational programs.  AQuA-RAT has provided the questions,
options, rationale, and the correct options.

languages:
english

tags:
math, qa, reasoning

paper:
https://arxiv.org/abs/1905.13319
"""

import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


_OPTION_PATTERN = re.compile(r"(?:^|,\s*)([a-e])\s*\)\s*(.*?)(?=,\s*[a-e]\s*\)\s*|$)", re.IGNORECASE)


def _mathqa_options(line) -> list[str]:
    """Read both the legacy split columns and the current ``options`` field."""
    legacy = [line.get(f"option_{label}") for label in "abcde"]
    if all(value is not None for value in legacy):
        return [str(value).strip() for value in legacy]

    raw_options = line.get("options")
    if isinstance(raw_options, str):
        matches = _OPTION_PATTERN.findall(raw_options)
        if len(matches) == 5:
            by_label = {label.lower(): value.strip() for label, value in matches}
            if all(label in by_label for label in "abcde"):
                return [by_label[label] for label in "abcde"]
    elif isinstance(raw_options, (list, tuple)) and len(raw_options) == 5:
        return [str(value).strip() for value in raw_options]

    raise KeyError("MathQA row has neither option_a..option_e nor five parseable options")


def mathqa_prompt(line, task_name: str = None):
    options = _mathqa_options(line)
    query = f"Problem: {line['Problem']}\n"
    query += "Options:\n"
    query += "".join(f"{key}) {choice}\n" for key, choice in zip("abcde", options))
    query += "Answer:"
    correct = re.search(r"[a-e]", str(line["correct"]).lower())
    if correct is None:
        raise ValueError(f"Invalid MathQA answer label: {line['correct']!r}")
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {choice}" for choice in options],
        gold_index="abcde".index(correct.group(0)),
    )


mathqa = LightevalTaskConfig(
    name="mathqa",
    prompt_function=mathqa_prompt,
    hf_repo="allenai/math_qa",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    mathqa,
]
