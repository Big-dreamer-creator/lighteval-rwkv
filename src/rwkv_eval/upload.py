from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import pyarrow.parquet as pq

from src.rwkv_eval.configs import RwkvModel, SamplingConfig


@dataclass
class Score:
    model: RwkvModel
    benchmark_name: str
    num_samples: int
    avg_k: float  # 整数 for 多次测量取平均; 小数 for 随机抽样
    score: float
    truncation_rate: float
    passed_details: list[Detail]
    wrong_details: list[Detail]
    failed_details: list[Detail]


@dataclass
class Detail:
    messages: list[dict[str, str]]  # openai 标准格式
    sampling_config: SamplingConfig
    answer: str
    ground_truth: str
    is_passed: bool


def get_score(file_path: str | Path) -> Score:
    """读取一个 benchmark 的 results JSON 及其详情。"""
    path = Path(file_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    task_names = [name for name in record["results"] if name != "all" and ":_average|" not in name]
    if len(task_names) != 1:
        raise ValueError(f"{path}: expected exactly one benchmark, found {len(task_names)}: {task_names}")
    (task_name,) = task_names
    if set(record["config_tasks"]) != {task_name}:
        raise ValueError(f"{path}: results and config_tasks must contain the same single benchmark")
    task_config = record["config_tasks"][task_name]
    task_metrics = record["results"][task_name]
    truncation_rate = task_metrics.get("truncation_rate")
    if (
        isinstance(truncation_rate, bool)
        or not isinstance(truncation_rate, (int, float))
        or not math.isfinite(truncation_rate)
        or not 0 <= truncation_rate <= 1
    ):
        raise ValueError(f"{path}: results must contain a finite truncation_rate in [0, 1]")
    metric_name = next(
        name
        for name in task_metrics
        if not name.endswith("_stderr")
        and name not in {"n_samples", "n_completions", "n_truncated", "truncation_rate"}
    )
    summary = record["summary_tasks"][task_name]
    num_samples = task_config["effective_num_docs"]
    avg_k = summary["n_completions"] / num_samples if summary["n_completions"] else 1
    model_name = record["config_general"]["model_config"]["model_name"]
    arch_version, data_version, param_size, _, ctx_len = Path(model_name).name.removesuffix(".pth").lower().split("-")
    results_root = next(parent for parent in path.parents if parent.name == "results")
    date_id = path.stem.removeprefix("results_")
    detail_path = (
        results_root.parent
        / "details"
        / path.parent.relative_to(results_root)
        / date_id
        / f"details_{task_name}_{date_id}.parquet"
    )
    passed_details, wrong_details, failed_details = get_eg_details(detail_path)
    return Score(
        model=RwkvModel(arch_version, data_version, param_size, int(ctx_len.removeprefix("ctx"))),
        benchmark_name=task_config["name"],
        num_samples=num_samples,
        avg_k=float(avg_k),
        score=float(task_metrics[metric_name]),
        truncation_rate=float(truncation_rate),
        passed_details=passed_details,
        wrong_details=wrong_details,
        failed_details=failed_details,
    )


def get_eg_details(file_path: str | Path) -> tuple[list[Detail], list[Detail], list[Detail]]:
    """读取单 benchmark 的三类样例；采样配置位于对应 results JSON 的同目录。"""
    path = Path(file_path)
    details_root = next(parent for parent in path.parents if parent.name == "details")
    sampling_config = get_sampling_config(
        details_root.parent
        / "results"
        / path.parent.parent.relative_to(details_root)
        / f"sampling_config_{path.parent.name}.json"
    )
    passed_details, wrong_details, failed_details = [], [], []
    buckets = (passed_details, wrong_details, failed_details)
    with pq.ParquetFile(path) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=64, columns=["doc", "model_response"]):
            for row in batch.to_pylist():
                doc, response = row["doc"], row["model_response"]
                choices = doc["choices"]
                gold_indices = doc["gold_index"] if isinstance(doc["gold_index"], list) else [doc["gold_index"]]
                golds = []
                for index in gold_indices:
                    gold = choices[index]
                    golds.extend(gold if isinstance(gold, list) else [gold])
                ground_truth = str(golds[0]) if len(golds) == 1 else json.dumps(golds, ensure_ascii=False)
                messages = (
                    response["input"]
                    if isinstance(response["input"], list)
                    else [{"role": "user", "content": response["input"] or doc["query"] or ""}]
                )
                if response["text"]:
                    scores = doc["specific"]["rwkv_rollout_scores"]
                    answers = doc["specific"]["rwkv_rollout_extracted_answers"]
                    finish_reasons = response["finish_reasons"] or ["stop"] * len(answers)
                else:
                    choice_scores = response["logprobs"][: len(choices)]
                    predicted_index = max(range(len(choice_scores)), key=choice_scores.__getitem__)
                    scores, answers, finish_reasons = (
                        [float(predicted_index in gold_indices)],
                        [choices[predicted_index]],
                        ["stop"],
                    )
                for score, answer, finish_reason in zip(scores, answers, finish_reasons, strict=True):
                    answer = str(answer)
                    outcome = 2 if finish_reason == "length" or not answer.strip() else int(score != 1)
                    if len(buckets[outcome]) < 20:
                        buckets[outcome].append(Detail(messages, sampling_config, answer, ground_truth, outcome == 0))
                if all(len(bucket) == 20 for bucket in buckets):
                    return buckets
    return buckets


def get_sampling_config(file_path: str | Path) -> SamplingConfig:
    """读取调度器保存的完整 SamplingConfig JSON；缺失字段或 null 均报错。"""
    sampling_config = SamplingConfig(**json.loads(Path(file_path).read_text(encoding="utf-8")))
    if any(value is None for value in vars(sampling_config).values()):
        raise ValueError(f"{file_path}: sampling configuration fields must not be null")
    return sampling_config


def upload(
    score: Score,
    *,
    field: Literal["knowledge", "reasoning", "maths", "coding", "instruction_following", "agentic", "vision"],
    cot_mode: Literal["NoCoT", "FakeCoT", "CoT"],
    token: str,
    api_url: str = "https://eval.rwkv.rs/test/api",
    sampling_config: SamplingConfig | None = None,
) -> dict[str, Any]:
    """Register the benchmark and upload its score to the Scoreboard API.

    ``api_url`` is the API root (use ``https://eval.rwkv.rs/api`` for production).
    All sampled details must share one sampling configuration; if there are no
    details, provide ``sampling_config`` explicitly.
    """
    details = score.passed_details + score.wrong_details + score.failed_details
    if sampling_config is None:
        if not details:
            raise ValueError("sampling_config is required when the score has no details")
        sampling_config = details[0].sampling_config
    if any(detail.sampling_config != sampling_config for detail in details):
        raise ValueError("all details must use the uploaded sampling_config")

    model = asdict(score.model)
    # Results filenames are parsed in lowercase, but the API requires G1a/G1i/etc.
    model["data_version"] = model["data_version"].capitalize()
    payload = {
        "model": model,
        "cot_mode": cot_mode,
        "sampling_config": asdict(sampling_config),
        "score": score.score,
        "truncation_rate": score.truncation_rate,
    }
    for name in ("passed_details", "wrong_details", "failed_details"):
        payload[name] = [
            {
                "messages": detail.messages,
                "answer": detail.answer,
                "ground_truth": detail.ground_truth,
                "is_passed": detail.is_passed,
            }
            for detail in getattr(score, name)
        ]

    with httpx.Client(
        base_url=f"{api_url.rstrip('/')}/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    ) as client:
        benchmark_response = client.post(
            "add_benchmarks",
            json={
                "name": score.benchmark_name,
                "field": field,
                "num_samples": score.num_samples,
                "avg_k": score.avg_k,
            },
        )
        benchmark_response.raise_for_status()
        payload["benchmark_id"] = benchmark_response.json()["benchmark_id"]
        response = client.post("upload", json=payload)
        response.raise_for_status()
        return response.json()
