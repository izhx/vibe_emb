from __future__ import annotations

import logging
import math
import time

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
        return control


class DatasetStatsCallback(TrainerCallback):
    """Report dataset consumption using Trainer's existing logging cadence."""

    def __init__(self, dataset: MultiDatasetBatchDataset) -> None:
        self.dataset = dataset

    def on_log(self, args, state, control, **kwargs):  # noqa: ANN001
        # Follow Trainer logging cadence instead of adding another interval.
        # This keeps dataset consumption stats aligned with loss/lr logs.
        if args.process_index == 0:
            self.dataset.log_consumption_stats("consumed")
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
