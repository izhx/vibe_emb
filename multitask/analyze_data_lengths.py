#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


DATASETS = [
    ("toolret", "data/train/toolret-train/toolret.jsonl"),
    ("msmarco-w-instructions", "data/train/msmarco-w-instructions/msmarco-w-instructions.jsonl"),
    ("codesearchnet", "data/train/codesearchnet/codesearchnet.jsonl"),
    ("fever", "data/train/embeddings-fine-tuning/fever.jsonl"),
    ("fiqa", "data/train/embeddings-fine-tuning/fiqa.jsonl"),
    ("hotpotqa", "data/train/embeddings-fine-tuning/hotpotqa.jsonl"),
    ("msmarco", "data/train/embeddings-fine-tuning/msmarco.jsonl"),
    ("nq", "data/train/embeddings-fine-tuning/nq.jsonl"),
    ("squadv2", "data/train/embeddings-fine-tuning/squadv2.jsonl"),
    ("trivia", "data/train/embeddings-fine-tuning/trivia.jsonl"),
]


def reservoir_sample(path: Path, limit: int, seed: int):
    rng = random.Random(seed)
    sample = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if len(sample) < limit:
                sample.append(line)
            else:
                j = rng.randrange(total)
                if j < limit:
                    sample[j] = line
    return total, [json.loads(x) for x in sample]


def quantiles(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.int64)
    return {
        "avg": float(arr.mean()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "over_256": float((arr > 256).mean()),
        "over_512": float((arr > 512).mean()),
    }


def encode_lengths(tokenizer, texts, batch_size):
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        out.extend(len(x) for x in encoded["input_ids"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/raw/Qwen2.5-0.5B")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("dataset\texamples\tsampled\tquery_avg\tquery_p95\tquery_p99\tquery_max\tquery_over256\tdoc_avg\tdoc_p95\tdoc_p99\tdoc_max\tdoc_over256")
    for idx, (name, rel_path) in enumerate(DATASETS):
        path = Path(rel_path)
        total, rows = reservoir_sample(path, args.sample_size, args.seed + idx)
        queries = []
        docs = []
        for row in rows:
            prompt = row.get("prompt") or "Given a query, retrieve passages that are relevant to the query."
            queries.append(f"<instruct>{prompt}\n<query>{row['query']}")
            if row.get("pos"):
                docs.append(row["pos"][0])
            for neg in row.get("neg", [])[:7]:
                docs.append(neg)
        query_stats = quantiles(encode_lengths(tokenizer, queries, args.batch_size))
        doc_stats = quantiles(encode_lengths(tokenizer, docs, args.batch_size))
        print(
            "\t".join(
                [
                    name,
                    str(total),
                    str(len(rows)),
                    f"{query_stats['avg']:.1f}",
                    str(query_stats["p95"]),
                    str(query_stats["p99"]),
                    str(query_stats["max"]),
                    f"{query_stats['over_256']:.3f}",
                    f"{doc_stats['avg']:.1f}",
                    str(doc_stats["p95"]),
                    str(doc_stats["p99"]),
                    str(doc_stats["max"]),
                    f"{doc_stats['over_256']:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
