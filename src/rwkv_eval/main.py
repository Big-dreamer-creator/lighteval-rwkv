# 读取 configs/benchmarks.toml 和 configs/models.toml
# 按照 benchmarks.toml 流式加载数据集到内存, 优先加载题目量少的, 优先加载无需 CoT 的.
# 每完成一个题目集加载, 按照 avg@k 构造请求队列, 按照 max_num_seqs 并发
# 高难度数据集 (如gpqa) 采用 ❀ 的 prompt template, 常规数据集使用 User Assistant 的 prompt template
# 答案提取器与判分器流式操作, 每道题拿到 is_passed 立即释放 prompts 和 completions 占用的内存
# 生成式选择题的作答 / gpqa 需要使用 albatross 提供的答案提取器, 并且学习它尽可能对模型作答进行兜底的做法.
# 中文题干的生成式选择题答案提取器的补丁在old分支.
# math500 同样参考 albatross.
# 答案提取器代码请在 src/rwkv_eval/answer_extract 完成.
# 请求采用五次重试策略, 若仍然报错立即终止
# 得到 Score 后立即调取 upload 完成上传
# 若上传失败, 重试五次, 仍然失败则存入上传失败池, 等待全部评估结束后再次上传

"""RWKV evaluation entry point.

The entry point only translates the RWKV TOML manifests into LightEval's
native model, pipeline, metrics, and tracker interfaces.  Backend scheduling,
prompt preparation, task loading, sampling counts, and scoring belong to
LightEval itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Sequence

from src.rwkv_eval.configs import BenchmarkSpec, ModelEndpoint, RwkvModel, SamplingConfig, read_benchmarks, read_models
from src.rwkv_eval.upload import Detail, Score
from src.rwkv_eval.upload import upload as upload_score


LOGGER = logging.getLogger(__name__)
REQUEST_RETRIES = 5
RETRY_DELAY = 1.0
DEFAULT_MAX_GENERATED_TOKENS = 8192

BenchmarkField = Literal[
    "knowledge",
    "reasoning",
    "maths",
    "coding",
    "instruction_following",
    "agentic",
    "vision",
]
CotMode = Literal["NoCoT", "FakeCoT", "CoT"]


class EvaluationError(RuntimeError):
    pass


def _prompt_template(selector: str, requested: str) -> str:
    if requested != "assistant":
        return requested
    name = selector.casefold()
    difficult = ("gpqa", "math", "aime", "olympiad", "code")
    return "bot" if any(keyword in name for keyword in difficult) else "assistant"


def _order_benchmarks(benchmarks: Sequence[BenchmarkSpec], max_samples: int | None) -> list[BenchmarkSpec]:
    """Order tasks without generation/CoT first, then sort by sample count."""
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.requests import SamplingMethod

    registry = Registry(tasks=",".join(spec.selector for spec in benchmarks), load_multilingual=True)
    tasks = registry.load_tasks()
    estimates: dict[str, tuple[int, int]] = {}
    for spec in benchmarks:
        selector = spec.selector.rsplit("|", 1)[0]
        matched = [
            task
            for task in tasks.values()
            if task.name == selector or task.name.startswith(f"{selector}:")
        ]
        sample_count = 0
        requires_cot = False
        for task in matched:
            try:
                LightevalTask.load_datasets({task.full_name: task}, 1)
                sample_count += sum(len(task.dataset[split]) for split in task.evaluation_split)
                requires_cot |= SamplingMethod.GENERATIVE in task.sampling_methods
            finally:
                task.dataset = None
                task._docs = None
                task._fewshot_docs = None
        if max_samples is not None:
            sample_count = min(sample_count, max_samples * max(len(matched), 1))
        estimates[spec.selector] = (sample_count, int(requires_cot))

    ordered = sorted(
        enumerate(benchmarks),
        key=lambda item: (
            estimates.get(item[1].selector, (10**18, 1))[1],
            estimates.get(item[1].selector, (10**18, 1))[0],
            item[0],
        ),
    )
    LOGGER.info(
        "benchmark order: %s",
        [f"{spec.selector}(samples={estimates.get(spec.selector, ('?', '?'))[0]})" for _, spec in ordered],
    )
    return [spec for _, spec in ordered]


def _sampling_config(cot_mode: CotMode, max_tokens: int, seed: int) -> SamplingConfig:
    values = {
        "NoCoT": (0.0, 0, 1.0, 0.0, 0.0, 1.0),
        "FakeCoT": (1.0, 32, 0.28, 0.0, 0.0, 1.0),
        "CoT": (0.96, 32, 0.76, 1.0, 0.1, 0.988),
    }
    temp, top_k, top_p, presence, frequency, decay = values[cot_mode]
    return SamplingConfig(max_tokens, temp, top_k, top_p, presence, frequency, decay, seed)


def _litelm_model(endpoint: ModelEndpoint, replicas: Sequence[ModelEndpoint], cot_mode: CotMode, template: str, max_tokens: int, seed: int):
    from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
    from lighteval.models.model_input import GenerationParameters

    if any(item.url != endpoint.url for item in replicas):
        raise ValueError("LiteLLM endpoint pooling requires replicas to share one base URL")
    sampling = _sampling_config(cot_mode, max_tokens, seed)
    return LiteLLMModelConfig(
        model_name=endpoint.model_name,
        provider="openai",
        base_url=f"{endpoint.url.rstrip('/')}/v1",
        api_key=endpoint.api_key,
        concurrent_requests=sum(item.max_num_seqs for item in replicas),
        max_model_length=endpoint.ctx_len,
        api_max_retry=5,
        generation_only=True,
        target_completions=4096,
        minimum_completions=3000,
        maximum_completions=6000,
        extra_body={
            "chat_template_kwargs": {
                "rwkv_prompt_template": template,
                "rwkv_generation_prompt": {
                    "NoCoT": "no_think",
                    "FakeCoT": "fake_think",
                    "CoT": "open_think",
                }[cot_mode],
            },
            "penalty_decay": sampling.penalty_decay,
        },
        generation_parameters=GenerationParameters(
            temperature=sampling.temp,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            max_new_tokens=sampling.max_generated_tokens,
            seed=sampling.seed,
        ),
    )


def _metric_value(value: Any) -> float:
    if isinstance(value, dict):
        value = next((item for item in value.values() if isinstance(item, (int, float))), 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _ground_truth(doc: Any) -> str:
    golds = [str(gold) for gold in doc.get_golds()]
    return golds[0] if len(golds) == 1 else str(golds)


def _detail_score(task: Any, doc: Any, response: Any, index: int) -> float:
    metric = next((item for item in task.metrics if str(getattr(item.category, "value", item.category)) == "GENERATIVE"), None)
    if metric is None:
        metric = task.metrics[0] if task.metrics else None
    if metric is None:
        return 0.0
    sample = response[index] if response.text and index < len(response.text) else response
    scorer = metric.sample_level_fn
    try:
        if hasattr(scorer, "compute_score"):
            return _metric_value(scorer.compute_score(doc, sample))
        if hasattr(scorer, "compute"):
            return _metric_value(scorer.compute(doc, sample))
        return _metric_value(metric.compute_sample(doc=doc, model_response=sample))
    except (IndexError, KeyError, TypeError, ValueError):
        return 0.0


def _native_score(pipeline: Any, task_name: str, metrics: dict[str, Any]) -> float:
    values = [value for name, value in metrics.items() if not name.endswith("_stderr")]
    if not values:
        return 0.0
    return _metric_value(values[0])


def _make_score(pipeline: Any, task_name: str, metrics: dict[str, Any], sampling: SamplingConfig) -> Score:
    task = pipeline.tasks_dict[task_name]
    native_details = pipeline.get_details().get(task_name, [])
    passed, wrong, failed = [], [], []
    truncated = total = 0
    for native in native_details:
        doc, response = native.doc, native.model_response
        messages = response.input if isinstance(response.input, list) else [{"role": "user", "content": str(response.input or doc.query)}]
        for index, answer in enumerate(response.final_text or [""]):
            finish_reason = response.finish_reasons[index] if index < len(response.finish_reasons) else "stop"
            is_truncated = finish_reason.lower() in {"length", "max_tokens"}
            truncated += int(is_truncated)
            total += 1
            detail = Detail(
                messages=[dict(message) for message in messages],
                sampling_config=sampling,
                answer=str(answer),
                ground_truth=_ground_truth(doc),
                is_passed=not is_truncated and bool(str(answer).strip()) and _detail_score(task, doc, response, index) == 1.0,
            )
            bucket = failed if is_truncated or not str(answer).strip() else passed if detail.is_passed else wrong
            if len(bucket) < 20:
                bucket.append(detail)
    avg_k = pipeline.task_avg_k.get(task_name, 1)
    return Score(
        model=RwkvModel(*_model_parts(pipeline.model.config.model_name, pipeline.model.config.max_model_length)),
        benchmark_name=task_name.split("|", 1)[0],
        num_samples=pipeline.task_sample_counts.get(task_name, len(native_details)),
        avg_k=float(avg_k),
        score=_native_score(pipeline, task_name, metrics),
        truncation_rate=(
            pipeline.task_truncation_counts.get(task_name, truncated)
            / pipeline.task_completion_counts.get(task_name, total)
            if pipeline.task_completion_counts.get(task_name, total)
            else 0.0
        ),
        passed_details=passed,
        wrong_details=wrong,
        failed_details=failed,
    )


def _model_parts(name: str, ctx_len: int) -> tuple[str, str, str, int]:
    parts = Path(name).name.removesuffix(".pth").lower().split("-")
    if len(parts) < 3 or parts[0] not in {"rwkv7", "rwkv7a", "rwkv7b"}:
        raise ValueError(f"cannot parse RWKV model name: {name}")
    if not re.fullmatch(r"g1[a-z0-9]*", parts[1]) or parts[2] not in {"1.5b", "2.9b", "7.2b", "13.3b"}:
        raise ValueError(f"invalid RWKV model name: {name}")
    return parts[0], parts[1].capitalize(), parts[2], ctx_len


def _field_for_task(task_name: str, benchmarks: Sequence[BenchmarkSpec]) -> BenchmarkField:
    leaf = task_name.split("|", 1)[0]
    for benchmark in benchmarks:
        selector = benchmark.selector.rsplit("|", 1)[-1]
        if leaf == selector or leaf.startswith(f"{selector}:"):
            return benchmark.field
    return benchmarks[0].field


def _score_path(score: Score, output_dir: str) -> Path:
    model = score.model
    model_id = f"{model.arch_version}-{model.data_version}-{model.param_size}-ctx{model.ctx_len}"
    benchmark_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", score.benchmark_name)
    path = Path(output_dir) / "scores" / model_id / f"{benchmark_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _persist_score(score: Score, field: BenchmarkField, output_dir: str, status: str) -> Path:
    path = _score_path(score, output_dir)
    payload = asdict(score)
    payload.update({"field": field, "upload_status": status})
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _set_score_status(path: Path, status: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["upload_status"] = status
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def _upload(score: Score, field: BenchmarkField, cot_mode: CotMode, token: str, url: str) -> None:
    last_error = None
    for attempt in range(REQUEST_RETRIES):
        try:
            await asyncio.to_thread(upload_score, score, field=field, cot_mode=cot_mode, token=token, api_url=url)
            return
        except Exception as error:
            last_error = error
            if attempt + 1 < REQUEST_RETRIES:
                await asyncio.sleep(RETRY_DELAY * 2**attempt)
    raise EvaluationError(f"score upload failed after {REQUEST_RETRIES} attempts") from last_error


async def evaluate(  # noqa: C901
    models: Sequence[ModelEndpoint],
    benchmarks: Sequence[BenchmarkSpec],
    *,
    cot_mode: CotMode = "CoT",
    prompt_template: str = "assistant",
    max_samples: int | None = None,
    max_generated_tokens: int = DEFAULT_MAX_GENERATED_TOKENS,
    seed: int = 42,
    scoreboard_token: str | None = None,
    scoreboard_url: str = "https://eval.rwkv.rs/test/api",
    no_upload: bool = False,
    output_dir: str = "results",
) -> tuple[list[Score], list[Score]]:
    if not models or not benchmarks:
        raise ValueError("models and benchmarks must not be empty")
    if not no_upload and not scoreboard_token:
        raise ValueError("scoreboard token is required unless --no-upload is used")

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    benchmarks = _order_benchmarks(benchmarks, max_samples)
    sampling = _sampling_config(cot_mode, max_generated_tokens, seed)
    successful: list[Score] = []
    pending: list[tuple[Score, BenchmarkField, Path]] = []
    groups: dict[tuple[str, int], list[ModelEndpoint]] = {}
    for endpoint in models:
        groups.setdefault((endpoint.model_name, endpoint.ctx_len), []).append(endpoint)

    for (model_name, ctx_len), replicas in groups.items():
        LOGGER.info("evaluating model=%s replicas=%d", model_name, len(replicas))
        for benchmark in benchmarks:
            LOGGER.info("starting LightEval for benchmark=%s", benchmark.selector)
            model_config = _litelm_model(
                replicas[0],
                replicas,
                cot_mode,
                _prompt_template(benchmark.selector, prompt_template),
                max_generated_tokens,
                seed,
            )
            tracker = EvaluationTracker(
                output_dir=output_dir,
                save_details=False,
                save_streaming_completions=True,
            )
            params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                max_samples=max_samples,
                load_tasks_multilingual=True,
                streaming_evaluation=True,
            )
            pipeline = await asyncio.to_thread(
                Pipeline, benchmark.selector, params, tracker, model_config=model_config
            )
            await asyncio.to_thread(pipeline.evaluate)
            await asyncio.to_thread(pipeline.show_results)
            result = pipeline.get_results()
            for task_name, metrics in result["results"].items():
                if task_name == "all" or ":_average|" in task_name:
                    continue
                score = _make_score(pipeline, task_name, metrics, sampling)
                successful.append(score)
                score_path = _persist_score(
                    score,
                    benchmark.field,
                    output_dir,
                    "local_only" if no_upload else "pending_upload",
                )
                if not no_upload:
                    try:
                        await _upload(
                            score,
                            benchmark.field,
                            cot_mode,
                            scoreboard_token or "",
                            scoreboard_url,
                        )
                    except EvaluationError:
                        pending.append((score, benchmark.field, score_path))
                    else:
                        _set_score_status(score_path, "uploaded")
                        score.passed_details.clear()
                        score.wrong_details.clear()
                        score.failed_details.clear()
            await asyncio.to_thread(pipeline.save_and_push_results)

    failures: list[Score] = []
    for score, field, score_path in pending:
        try:
            await _upload(score, field, cot_mode, scoreboard_token or "", scoreboard_url)
        except EvaluationError:
            _set_score_status(score_path, "upload_failed")
            failures.append(score)
        else:
            _set_score_status(score_path, "uploaded")
            score.passed_details.clear()
            score.wrong_details.clear()
            score.failed_details.clear()
    return successful, failures


def _argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run RWKV through the LightEval pipeline.")
    parser.add_argument("--models", type=Path, default=root / "configs/models.toml")
    parser.add_argument("--benchmarks", type=Path, default=root / "configs/benchmarks.toml")
    parser.add_argument("--cot-mode", choices=("NoCoT", "FakeCoT", "CoT"), default="CoT")
    parser.add_argument("--prompt-template", choices=("bot", "assistant", "function_calling"), default="assistant")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-generated-tokens", type=int, default=DEFAULT_MAX_GENERATED_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--scoreboard-token", default=os.environ.get("SCOREBOARD_API_TOKEN"))
    parser.add_argument("--scoreboard-url", default=os.environ.get("SCOREBOARD_API_URL", "https://eval.rwkv.rs/test/api"))
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.max_generated_tokens <= 0:
        raise SystemExit("--max-generated-tokens must be positive")
    scores, failures = asyncio.run(
        evaluate(
            read_models(args.models),
            read_benchmarks(args.benchmarks),
            cot_mode=args.cot_mode,
            prompt_template=args.prompt_template,
            max_samples=args.max_samples,
            max_generated_tokens=args.max_generated_tokens,
            seed=args.seed,
            scoreboard_token=args.scoreboard_token,
            scoreboard_url=args.scoreboard_url,
            no_upload=args.no_upload,
            output_dir=args.output_dir,
        )
    )
    if failures:
        LOGGER.error("%d score uploads remain pending", len(failures))
        return 1
    LOGGER.info("completed %d scores", len(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
