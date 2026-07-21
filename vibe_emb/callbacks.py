from __future__ import annotations

import json
import logging
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.trainer_callback import TrainerCallback

from .data import MultiDatasetBatchDataset

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class DatasetRefreshCallback(TrainerCallback):
    """Rebuild the deterministic global batch plan at each epoch boundary."""

    def __init__(self, dataset: MultiDatasetBatchDataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        epoch = int(state.epoch or 0)
        # The dataset already owns epoch 0 when Trainer starts. Rebuild only
        # when Trainer advances to a new epoch.
        if epoch != self.dataset.epoch:
            self.dataset.refresh_epoch(epoch)
            logger.info(
                "Refreshed dataset batch plan for epoch %d; non-persistent DataLoader workers "
                "will be created from this epoch state.",
                epoch,
            )
        return control


class DatasetStatsCallback(TrainerCallback):
    """Report main-process consumption and DataLoader wait telemetry."""

    def __init__(self, dataset: MultiDatasetBatchDataset, trainer: Any) -> None:
        self.dataset = dataset
        self.trainer = trainer
        self._metrics_handle = None
        self._last_log_time: float | None = None
        self._last_log_step = 0
        self._peak_rss_bytes = 0
        self._rss_scope = "rank_process"
        self._rows_since_flush = 0

    def _metrics_path(self, args) -> Path:  # noqa: ANN001
        return Path(args.output_dir) / f"dataloader_metrics_rank_{args.process_index}.jsonl"

    def _open_metrics(self, args) -> None:  # noqa: ANN001
        if self._metrics_handle is not None:
            return
        path = self._metrics_path(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A process summary and its JSONL must describe the same invocation.
        # Resume evidence that spans invocations is copied/aggregated by the
        # caller rather than mixing stale rows in a reused output directory.
        self._metrics_handle = path.open("w", encoding="utf-8")

    def _sample_rss(self) -> int:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss
            rss += sum(child.memory_info().rss for child in process.children(recursive=True) if child.is_running())
            self._rss_scope = "rank_process_tree"
        except Exception:
            # Linux ru_maxrss is KiB and records the rank process peak. This
            # fallback keeps telemetry available without making psutil a hard
            # training dependency.
            rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            self._rss_scope = "rank_process"
        self._peak_rss_bytes = max(self._peak_rss_bytes, int(rss))
        return int(rss)

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        del kwargs
        self._open_metrics(args)
        self._last_log_time = time.monotonic()
        self._last_log_step = state.global_step
        self._sample_rss()
        return control

    def on_log(self, args, state, control, **kwargs):  # noqa: ANN001
        del kwargs
        self._open_metrics(args)
        now = time.monotonic()
        elapsed = max(0.0, now - (self._last_log_time or now))
        step_delta = max(0, state.global_step - self._last_log_step)
        wait = self.trainer.dataloader_wait_tracker.snapshot_window(reset=True)
        # Trainer emits a second on_log event for final aggregate metrics at
        # the same global step. It has no new DataLoader sample and must not
        # become a duplicate/zero-count benchmark row.
        if step_delta == 0 and wait["wait_count"] == 0:
            return control
        totals = self.dataset.consumption_totals()
        row = {
            "rank": args.process_index,
            "global_step": state.global_step,
            "epoch": float(state.epoch or 0.0),
            **wait,
            **totals,
            "step_count": step_delta,
            "elapsed_seconds": elapsed,
            "mean_step_time_ms": 1000.0 * elapsed / step_delta if step_delta else 0.0,
            "queries_per_second": (
                wait["local_instances"] * args.world_size / elapsed if elapsed > 0 else 0.0
            ),
            "rss_bytes": self._sample_rss(),
        }
        assert self._metrics_handle is not None
        self._metrics_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._rows_since_flush += 1
        if self._rows_since_flush >= 100:
            self._metrics_handle.flush()
            self._rows_since_flush = 0

        # Follow Trainer logging cadence instead of adding another interval.
        # This keeps dataset consumption stats aligned with loss/lr logs.
        if args.process_index == 0:
            logger.info(
                "Dataset consumed totals: epoch=%s rank=%s/%s "
                "consumed_local_batches=%d consumed_local_instances=%d",
                self.dataset.epoch,
                args.process_index,
                args.world_size,
                totals["consumed_local_batches"],
                totals["consumed_local_instances"],
            )
            logger.info(
                "DataLoader wait: step=%d window_count=%d mean=%.3f ms p95=%.3f ms",
                state.global_step,
                wait["wait_count"],
                wait["mean_batch_wait_ms"],
                wait["p95_batch_wait_ms"],
            )
        self._last_log_time = now
        self._last_log_step = state.global_step
        return control

    def finalize(self, args, state) -> None:  # noqa: ANN001
        if self._metrics_handle is not None:
            self._metrics_handle.flush()
            self._metrics_handle.close()
            self._metrics_handle = None
            self._rows_since_flush = 0
        dataloader_info = self.trainer.train_dataloader_info
        self._sample_rss()
        summary = {
            "rank": args.process_index,
            "global_step": state.global_step,
            "epoch": float(state.epoch or 0.0),
            **self.trainer.dataloader_wait_tracker.snapshot_total(),
            **self.dataset.consumption_totals(),
            "peak_rss_bytes": self._peak_rss_bytes,
            "rss_scope": self._rss_scope,
            "environment": {
                "hostname": platform.node(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }
        if dataloader_info is not None:
            summary["dataloader"] = dict(dataloader_info)
        path = Path(args.output_dir) / f"dataloader_metrics_rank_{args.process_index}_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del kwargs
        self.finalize(args, state)
        return control


class TrainingProgressCallback(TrainerCallback):
    """Emit newline-delimited progress and ETA logs for non-interactive log viewers."""

    def __init__(self, smoothing: float = 0.2) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in the interval (0, 1].")
        self.smoothing = smoothing
        self._start_time: float | None = None
        self._last_time: float | None = None
        self._last_step = 0
        self._smoothed_seconds_per_step: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self._start_time = time.monotonic()
        self._last_time = self._start_time
        self._last_step = state.global_step
        self._smoothed_seconds_per_step = None
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if args.process_index != 0:
            return control

        logging_steps = args.logging_steps
        if 0 < logging_steps < 1:
            logging_steps = math.ceil(state.max_steps * logging_steps)
        logging_steps = max(1, int(logging_steps))
        if state.global_step % logging_steps != 0 and state.global_step != state.max_steps:
            return control

        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            self._last_step = state.global_step
            return control

        completed_steps = state.global_step - self._last_step
        if completed_steps <= 0:
            return control

        seconds_per_step = max((now - self._last_time) / completed_steps, 1e-12)
        if self._smoothed_seconds_per_step is None:
            self._smoothed_seconds_per_step = seconds_per_step
        else:
            self._smoothed_seconds_per_step = (
                self.smoothing * seconds_per_step
                + (1 - self.smoothing) * self._smoothed_seconds_per_step
            )

        remaining_steps = max(0, state.max_steps - state.global_step)
        eta_seconds = remaining_steps * self._smoothed_seconds_per_step
        elapsed_seconds = now - (self._start_time or now)
        progress = 100 * state.global_step / state.max_steps if state.max_steps > 0 else 0.0
        logger.info(
            "Training progress: step=%d/%d, progress=%.2f%%, elapsed=%s, speed=%.3f steps/s, eta=%s",
            state.global_step,
            state.max_steps,
            progress,
            _format_duration(elapsed_seconds),
            1 / self._smoothed_seconds_per_step,
            _format_duration(eta_seconds),
        )

        self._last_time = now
        self._last_step = state.global_step
        return control
