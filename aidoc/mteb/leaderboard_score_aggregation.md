# Local MTEB benchmark summary

The repository utility is a local-directory version of MTEB's documented
leaderboard API. MTEB performs result loading, benchmark filtering, aggregation,
ranking, and DataFrame construction:

```python
benchmark = mteb.get_benchmark(benchmark_name)
results = cache.load_results(
    tasks=benchmark,
    require_model_meta=True,
    include_remote=False,
)
benchmark_scores_df = results.get_benchmark_result()
```

The only repository-specific component is `LocalResultCache`. Instead of calling
`ResultCache.download_from_remote()`, it supplies the task JSON files found in the
revision directories passed through `--results`. It does not implement score or
task-type aggregation.

Each result directory must be an MTEB model revision directory containing
`model_meta.json` and task result JSON files. Metadata for an unregistered local
model is registered only in the current Python process so MTEB's summary table
keeps that model row.

## Usage

One model and one benchmark:

```bash
/mnt/share/envs/embt/bin/python scripts/mteb/summarize_benchmark.py \
  --results results/mteb_eval/models__F2LLM-v2-0.6B/no_revision_available \
  --benchmark 'MTEB(cmn, v1)'
```

Multiple models and benchmarks:

```bash
/mnt/share/envs/embt/bin/python scripts/mteb/summarize_benchmark.py \
  --results \
    results/mteb_offical/model-a/revision-a \
    results/mteb_eval/model-b/revision-b \
  --benchmark 'MTEB(eng, v2)' \
  --benchmark 'MTEB(Code, v1)' \
  --benchmark 'MTEB(cmn, v1)'
```

Output contains one block per benchmark. Each block is the DataFrame returned by
MTEB: models are rows and aggregate/task-type metrics are columns. Values retain
MTEB's native 0-1 scale. The command does not download or update remote results.

Use `--format csv` for CSV output. It preserves the table layout: every benchmark
is a separate block containing benchmark metadata, its own header, and model rows;
blocks are separated by an empty line:

```bash
/mnt/share/envs/embt/bin/python scripts/mteb/summarize_benchmark.py \
  --results results/mteb_eval/models__F2LLM-v2-0.6B/no_revision_available \
  --benchmark 'MTEB(Code, v1)' \
  --benchmark 'MTEB(cmn, v1)' \
  --format csv > benchmark_summary.csv
```

Add `--per-task` to append MTEB's per-task leaderboard table after the summary
inside every benchmark block. This works with both `table` and `csv` output. In
CSV output, `Result type,Summary` and `Result type,Per task` identify the two
tables while benchmarks remain separated into blocks:

```bash
/mnt/share/envs/embt/bin/python scripts/mteb/summarize_benchmark.py \
  --results results/mteb_eval/models__F2LLM-v2-0.6B/no_revision_available \
  --benchmark 'MTEB(Code, v1)' \
  --benchmark 'MTEB(cmn, v1)' \
  --format csv \
  --per-task > benchmark_with_tasks.csv
```
