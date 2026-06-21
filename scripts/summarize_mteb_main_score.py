#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


BASIC_TASKS = [
    "NanoArguAnaRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval",
    "NanoNFCorpusRetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
    "NanoSciFactRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoTouche2020Retrieval",
]

CODE_TASKS = [
    "AppsRetrieval",
    "CodeEditSearchRetrieval",
    "CodeSearchNetRetrieval",
    "CodeTransOceanContest",
    "CodeTransOceanDL",
    "CosQA",
    "StackOverflowQA",
]

INSTRUCT_TASKS = [
    "Core17InstructionRetrieval",
    "News21InstructionRetrieval",
    "Robust04InstructionRetrieval",
]

REASONING_TASKS = [
    "BrightBiologyRetrieval",
    "BrightEconomicsRetrieval",
    "BrightPsychologyRetrieval",
    "BrightRoboticsRetrieval",
    "BrightSustainableLivingRetrieval",
    "BrightTheoremQATheoremsRetrieval",
]

TOOL_TASK = "ToolRetRetrieval"
TOOL_SUBSETS = ["web", "code", "customized"]
MEAN_SUBSET_TASKS = {"CodeEditSearchRetrieval", "CodeSearchNetRetrieval"}

CATEGORIES = [
    ("basic", BASIC_TASKS),
    ("code", CODE_TASKS),
    ("instruct", INSTRUCT_TASKS),
    ("reasoning", REASONING_TASKS),
]


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def load_result(result_dir: Path, task_name: str) -> dict | None:
    path = result_dir / f"{task_name}.json"
    if not path.exists():
        warn(f"missing result file: {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warn(f"failed to read {path}: {exc}")
        return None


def iter_score_entries(result: dict):
    scores = result.get("scores")
    if not isinstance(scores, dict):
        return

    for split_entries in scores.values():
        if isinstance(split_entries, list):
            for entry in split_entries:
                if isinstance(entry, dict):
                    yield entry
        elif isinstance(split_entries, dict):
            yield split_entries


def get_single_score_entry(result: dict, task_name: str) -> dict:
    scores = result["scores"]
    assert isinstance(scores, dict), f"invalid scores: {task_name}"
    assert len(scores) == 1, f"expected one score split: {task_name}"

    split_entries = next(iter(scores.values()))
    assert isinstance(split_entries, list), f"expected score entries list: {task_name}"
    assert len(split_entries) == 1, f"expected one score entry: {task_name}"
    assert isinstance(split_entries[0], dict), f"invalid score entry: {task_name}"
    return split_entries[0]


def read_mean_subset_task(result_dir: Path, task_name: str) -> tuple[float, float]:
    result = load_result(result_dir, task_name)
    if result is None:
        return 0.0, 0.0

    entries = list(iter_score_entries(result))
    assert entries, f"missing score entries: {task_name}"
    assert all("main_score" in entry for entry in entries), f"missing main_score: {task_name}"

    score = mean([float(entry["main_score"]) for entry in entries])
    entry_times = [entry.get("evaluation_time") for entry in entries]
    if all(eval_time is not None for eval_time in entry_times):
        eval_time = sum(float(eval_time) for eval_time in entry_times)
    else:
        try:
            eval_time = float(result.get("evaluation_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            warn(f"invalid evaluation_time: {task_name}")
            eval_time = 0.0

    return score, eval_time


def read_single_task(result_dir: Path, task_name: str) -> tuple[float, float]:
    if task_name in MEAN_SUBSET_TASKS:
        return read_mean_subset_task(result_dir, task_name)

    result = load_result(result_dir, task_name)
    if result is None:
        return 0.0, 0.0

    entry = get_single_score_entry(result, task_name)
    assert "main_score" in entry, f"missing main_score: {task_name}"
    score = float(entry["main_score"])

    try:
        eval_time = float(result.get("evaluation_time", 0.0) or 0.0)
    except (TypeError, ValueError):
        warn(f"invalid evaluation_time: {task_name}")
        eval_time = 0.0

    return score, eval_time


def read_tool_task(result_dir: Path) -> tuple[list[float], list[float]]:
    result = load_result(result_dir, TOOL_TASK)
    if result is None:
        return [0.0] * len(TOOL_SUBSETS), [0.0] * len(TOOL_SUBSETS)

    scores_by_subset = {}
    for entry in iter_score_entries(result):
        subset = entry.get("hf_subset")
        if subset in TOOL_SUBSETS and "main_score" in entry:
            scores_by_subset[subset] = float(entry["main_score"])

    scores = []
    for subset in TOOL_SUBSETS:
        if subset not in scores_by_subset:
            warn(f"missing main_score: {TOOL_TASK}/{subset}")
        scores.append(scores_by_subset.get(subset, 0.0))

    try:
        eval_time = float(result.get("evaluation_time", 0.0) or 0.0)
    except (TypeError, ValueError):
        warn(f"invalid evaluation_time: {TOOL_TASK}")
        eval_time = 0.0

    return scores, [eval_time, 0.0, 0.0]


def fmt_score(value: float) -> str:
    return f"{value * 100:.3f}"


def fmt_time(value: float) -> str:
    return f"{value:.0f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize MTEB main_score and evaluation_time from a result directory."
    )
    parser.add_argument("result_dir", help="Path to a MTEB result directory")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    print(result_dir)

    task_scores = {}
    task_times = {}
    for _, tasks in CATEGORIES:
        for task_name in tasks:
            score, eval_time = read_single_task(result_dir, task_name)
            task_scores[task_name] = score
            task_times[task_name] = eval_time

    tool_scores, tool_times = read_tool_task(result_dir)

    category_scores = []
    category_times = []
    for _, tasks in CATEGORIES:
        category_scores.append(mean([task_scores[task_name] for task_name in tasks]))
        category_times.append(sum(task_times[task_name] for task_name in tasks))

    category_scores.append(mean(tool_scores))
    category_times.append(sum(tool_times))

    task_headers = [
        task_name for _, tasks in CATEGORIES for task_name in tasks
    ] + [f"{TOOL_TASK}-{subset}" for subset in TOOL_SUBSETS]
    task_score_values = [
        task_scores[task_name] for _, tasks in CATEGORIES for task_name in tasks
    ] + tool_scores
    task_time_values = [
        task_times[task_name] for _, tasks in CATEGORIES for task_name in tasks
    ] + tool_times

    print(",".join(["basic", "code", "instruct", "reasoning", "tool", "", ""] + task_headers))
    print(",".join([fmt_score(value) for value in category_scores] + ["", ""] + [fmt_score(value) for value in task_score_values]))
    print(",".join([fmt_time(value) for value in category_times] + ["", ""] + [fmt_time(value) for value in task_time_values]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
