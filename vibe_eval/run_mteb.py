from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    format="%(asctime)s|%(name)s:%(lineno)s|%(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)

os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'

DEFAULT_TASKS = [
    "NanoMSMARCORetrieval",
    "AppsRetrieval",
    "Core17InstructionRetrieval",

    # # basic
    # "NanoArguAnaRetrieval",
    # "NanoClimateFeverRetrieval",
    # "NanoDBPediaRetrieval",
    # "NanoFEVERRetrieval",
    # "NanoFiQA2018Retrieval",
    # "NanoHotpotQARetrieval",
    # "NanoMSMARCORetrieval",
    # "NanoNFCorpusRetrieval",
    # "NanoNQRetrieval",
    # "NanoQuoraRetrieval",
    # "NanoSciFactRetrieval",
    # "NanoSCIDOCSRetrieval",
    # "NanoTouche2020Retrieval",

    # # code
    # "AppsRetrieval",
    # "CodeEditSearchRetrieval",
    # "CodeSearchNetRetrieval",
    # "CodeTransOceanContest",
    # "CodeTransOceanDL",
    # "CosQA",
    # "StackOverflowQA",

    # # FollowIR
    # "Core17InstructionRetrieval",
    # "News21InstructionRetrieval",
    # "Robust04InstructionRetrieval",

    # # BRIGHT
    # "BrightBiologyRetrieval",
    # "BrightEconomicsRetrieval",
    # "BrightPsychologyRetrieval",
    # "BrightRoboticsRetrieval",
    # "BrightSustainableLivingRetrieval",
    # "BrightTheoremQATheoremsRetrieval",

    # "ToolRetRetrieval"
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local decoder-only embedding model checkpoint with MTEB."
    )
    parser.add_argument(
        "--model_name_or_path",
        "--checkpoint",
        dest="model_name_or_path",
        # default="data/raw/Qwen2.5-0.5B",
        default="/mnt/share/models/Qwen3-0.6B",
        help=(
            "Full Hugging Face model checkpoint/directory to evaluate, or the base model "
            "when --adapter is set. --checkpoint is kept as a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT adapter checkpoint to load and merge onto --model_name_or_path.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Explicit MTEB task names. When neither --tasks nor --benchmarks is "
            "provided, the script uses its built-in default task list."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "MTEB benchmark names, for example 'MTEB(eng, v2)' "
            "'MTEB(Code, v1)' BRIGHT. Benchmark-defined tasks, subsets, and "
            "splits are used."
        ),
    )
    parser.add_argument(
        "--no_model",
        "--no-model",
        dest="no_model",
        action="store_true",
        help=(
            "Do not load a tokenizer or model and do not evaluate. Resolve the "
            "selected tasks, call task.load_data() for each, and exit."
        ),
    )
    # parser.add_argument("--cache_dir", default="/data8/zhangxin/vibe_emb/.cache")
    parser.add_argument("--cache_dir", default="/mnt/share/emb/mteb_cache")
    parser.add_argument(
        "--output_folder",
        default="results/mteb_eval",
        help=(
            "Result directory. JSON files are stored below its "
            "<model>/<revision>/ directory."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default='cuda')
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument(
        "--query_instruction",
        default="Given a question, retrieve passages that can help answer the question.",
    )
    parser.add_argument(
        "--query_instruction_format",
        default="Instruct: {instruction}\nQuery: ",
    )
    parser.add_argument(
        "--no_task_prompts",
        action="store_true",
        help="Ignore MTEB task-specific query prompts and always use --query_instruction.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Always rerun selected evaluations and overwrite cached results.",
    )
    parser.add_argument("--verbosity", type=int, choices=range(4), default=2)
    return parser.parse_args()


def resolve_task_selection(
    task_names: Sequence[str] | None,
    benchmark_names: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve CLI defaults without mixing default tasks into benchmark runs."""
    if task_names is None and not benchmark_names:
        return list(DEFAULT_TASKS), []
    return list(task_names or ()), list(benchmark_names or ())


def build_tasks(
    task_names: Sequence[str] | None = None,
    benchmark_names: Sequence[str] | None = None,
) -> list[Any]:
    import mteb

    requested_tasks, requested_benchmarks = resolve_task_selection(
        task_names,
        benchmark_names,
    )

    tasks: list[Any] = []
    local_task_names = {"ToolRetRetrieval"}
    if local_task_names.intersection(requested_tasks):
        from vibe_eval.tasks.toolret import ToolRetRetrieval

        local_tasks = {
            "ToolRetRetrieval": ToolRetRetrieval,
        }
        tasks.extend(
            local_tasks[name]() for name in requested_tasks if name in local_tasks
        )

    mteb_task_names = [
        name for name in requested_tasks if name not in local_task_names
    ]
    if mteb_task_names:
        tasks.extend(mteb.get_tasks(tasks=mteb_task_names))

    for benchmark_name in requested_benchmarks:
        benchmark = mteb.get_benchmark(benchmark_name)
        tasks.extend(benchmark.tasks)

    if not tasks:
        raise ValueError("No MTEB tasks were selected.")

    from vibe_eval.tasks.belebele_retrieval import patch_belebele_tasks
    from vibe_eval.tasks.code_edit_search_retrieval import patch_code_edit_search_tasks
    from vibe_eval.tasks.mind_small_reranking_patch import patch_mind_small_reranking_tasks

    tasks = patch_code_edit_search_tasks(tasks)
    tasks = patch_belebele_tasks(tasks)
    return patch_mind_small_reranking_tasks(tasks)


def configure_mteb_verbosity(verbosity: int) -> None:
    levels = {
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    logging.getLogger("mteb").setLevel(levels[verbosity])


def _iter_download_tasks(tasks: Sequence[Any]):
    for task in tasks:
        if getattr(task, "is_aggregate", False):
            yield from _iter_download_tasks(task.tasks)
        else:
            yield task


def _unload_task_data(task: Any) -> None:
    if getattr(task, "data_loaded", False):
        task.unload_data()

    # Some legacy retrieval tasks keep their loaded data outside `task.dataset`.
    for attribute in (
        "corpus",
        "queries",
        "relevant_docs",
        "instructions",
        "top_ranked",
    ):
        if attribute in vars(task):
            setattr(task, attribute, None)
    gc.collect()


def download_task_data(tasks: Sequence[Any]) -> None:
    download_tasks = list(_iter_download_tasks(tasks))
    total = len(download_tasks)
    failures: list[tuple[str, Exception]] = []
    for index, task in enumerate(download_tasks, start=1):
        task_name = task.metadata.name
        print(f"[{index}/{total}] Loading dataset for {task_name}", flush=True)
        try:
            task.load_data()
        except Exception as error:
            failures.append((task_name, error))
            print(
                f"[{index}/{total}] Failed dataset for {task_name}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        finally:
            _unload_task_data(task)
        print(f"[{index}/{total}] Cached dataset for {task_name}", flush=True)

    if failures:
        failed_names = ", ".join(name for name, _ in failures)
        raise RuntimeError(
            f"Failed to cache {len(failures)}/{total} datasets: {failed_names}"
        )


def main() -> None:
    args = parse_args()

    root = ROOT
    os.chdir(root)

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    os.environ["HF_HOME"] = str(cache_dir.resolve())
    # os.environ["HF_DATASETS_CACHE"] = str((cache_dir / "datasets").resolve())
    # os.environ["TRANSFORMERS_CACHE"] = str((cache_dir / "transformers").resolve())

    tasks = build_tasks(args.tasks, args.benchmarks)
    if args.no_model:
        download_task_data(tasks)
        return

    import mteb
    from vibe_eval.modeling import QwenDecoderOnlyEmbedder
    from vibe_eval.mteb_patches import (
        install_query_dataloader_patch,
        install_retrieval_qrels_offline_patch,
        install_reranking_top_ranked_patch,
        create_result_cache,
    )

    configure_mteb_verbosity(args.verbosity)
    install_query_dataloader_patch()
    install_retrieval_qrels_offline_patch()

    install_reranking_top_ranked_patch(tasks)

    model = QwenDecoderOnlyEmbedder(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        query_instruction=args.query_instruction,
        query_instruction_format=args.query_instruction_format,
        use_task_prompts=not args.no_task_prompts,
        trust_remote_code=args.trust_remote_code,
        use_flash_attn=not args.no_flash_attn,
    )

    result_cache = create_result_cache(mteb, args.output_folder)
    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=result_cache,
        co2_tracker=False,
        overwrite_strategy="always" if args.overwrite_results else "only-missing",
        encode_kwargs={"batch_size": args.batch_size},
    )
    # print(
    #     json.dumps(
    #         [result.model_dump(mode="json") for result in results.task_results],
    #         ensure_ascii=False,
    #         indent=2,
    #     )
    # )
    logging.warning('done')


if __name__ == "__main__":
    main()
