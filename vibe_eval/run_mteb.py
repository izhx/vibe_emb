from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
logging.getLogger("mteb").setLevel(logging.INFO)

os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'

DEFAULT_TASKS = [
    # "NanoMSMARCORetrieval",
    # "AppsRetrieval",
    # "Core17InstructionRetrieval",

    # basic
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

    # code
    "AppsRetrieval",
    "CodeEditSearchRetrieval",
    "CodeSearchNetRetrieval",
    "CodeTransOceanContest",
    "CodeTransOceanDL",
    "CosQA",
    "StackOverflowQA",

    # FollowIR
    "Core17InstructionRetrieval",
    "News21InstructionRetrieval",
    "Robust04InstructionRetrieval",

    # # BRIGHT
    # "BrightBiologyRetrieval",
    # "BrightEconomicsRetrieval",
    # "BrightPsychologyRetrieval",
    # "BrightRoboticsRetrieval",
    # "BrightSustainableLivingRetrieval",
    # "BrightTheoremQATheoremsRetrieval",

    "ToolRetRetrieval"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local decoder-only embedding model checkpoint with MTEB."
    )
    parser.add_argument(
        "--model_name_or_path",
        "--checkpoint",
        dest="model_name_or_path",
        default="data/raw/Qwen2.5-0.5B",
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
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--eval_splits", nargs="+", default=None)
    parser.add_argument("--eval_subsets", nargs="+", default=None)
    parser.add_argument("--cache_dir", default="/data8/zhangxin/vibe_emb/.cache")
    parser.add_argument("--output_folder", default="results/mteb_eval")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--query_max_length", type=int, default=2048)
    parser.add_argument("--corpus_max_length", type=int, default=2048)
    parser.add_argument("--device", default='cuda')
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument(
        "--query_instruction",
        default="Given a query, retrieve passages that are relevant to the query.",
    )
    parser.add_argument(
        "--query_instruction_format",
        default="Instruct: {}\nQuery: {}",
    )
    parser.add_argument(
        "--no_task_prompts",
        action="store_true",
        help="Ignore MTEB task-specific query prompts and always use --query_instruction.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--overwrite_results", action="store_true")
    parser.add_argument("--verbosity", type=int, default=2)
    return parser.parse_args()


def build_tasks(task_names: list[str]):
    import mteb
    from vibe_eval.tasks.toolret import ToolRetRetrieval

    local_tasks = {
        "ToolRetRetrieval": ToolRetRetrieval,
    }
    tasks = [local_tasks[name]() for name in task_names if name in local_tasks]
    mteb_task_names = [name for name in task_names if name not in local_tasks]
    if mteb_task_names:
        tasks.extend(mteb.get_tasks(tasks=mteb_task_names))
    return tasks


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

    import mteb
    from vibe_eval.modeling import QwenDecoderOnlyEmbedder

    model = QwenDecoderOnlyEmbedder(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        query_max_length=args.query_max_length,
        corpus_max_length=args.corpus_max_length,
        query_instruction=args.query_instruction,
        query_instruction_format=args.query_instruction_format,
        use_task_prompts=not args.no_task_prompts,
        trust_remote_code=args.trust_remote_code,
        use_flash_attn=not args.no_flash_attn,
    )

    tasks = build_tasks(args.tasks)
    evaluator = mteb.MTEB(tasks=tasks)
    results = evaluator.run(
        model,
        output_folder=args.output_folder,
        eval_splits=args.eval_splits,
        eval_subsets=args.eval_subsets,
        overwrite_results=args.overwrite_results,
        verbosity=args.verbosity,
        encode_kwargs={"batch_size": args.batch_size, "show_progress_bar": True},
    )
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    return


if __name__ == "__main__":
    main()
