from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .arguments import EmbedTrainingExtras, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingOutput(ModelOutput):
    """Trainer-compatible loss output with optional embeddings for inspection."""
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None


def _dtype_from_name(name: Optional[str], bf16: bool = False, fp16: bool = False):
    if name:
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16"}:
            return torch.float16
        if name in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype: {name}")
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def build_base_model(model_args: ModelConfig, bf16: bool = False, fp16: bool = False) -> PreTrainedModel:
    torch_dtype = _dtype_from_name(model_args.torch_dtype, bf16=bf16, fp16=fp16)
    kwargs = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return AutoModel.from_pretrained(model_args.model_name_or_path, **kwargs)


def maybe_apply_peft(base_model: PreTrainedModel, model_args: ModelConfig) -> PreTrainedModel:
    if model_args.peft_adapter_name_or_path:
        adapter_path = model_args.peft_adapter_name_or_path
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.isdir(adapter_path) or not os.path.isfile(adapter_config_path):
            raise ValueError(
                "model.peft_adapter_name_or_path must point to a PEFT adapter checkpoint directory "
                f"containing adapter_config.json: {adapter_path}"
            )
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("model.peft_adapter_name_or_path requires `peft` to be installed.") from exc

        if model_args.peft_config:
            logger.info(
                "Both peft_adapter_name_or_path and peft_config are set; loading adapter config from %s "
                "and keeping peft_config only as experiment metadata.",
                adapter_path,
            )
        # This is a warm-start path, not a Trainer checkpoint resume. The base
        # model still comes from model_name_or_path, while adapter weights and
        # their PEFT config come from the adapter checkpoint and remain trainable.
        # PEFT infers bare "cuda" by default when CUDA is available. With
        # safetensors that can create a context on visible cuda:0 for every
        # rank before Trainer moves the model to each local rank. Keep adapter
        # warm-start loading on CPU and let Trainer/DDP perform the device move.
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True, torch_device="cpu")
        model.print_trainable_parameters()
        return model

    if not model_args.peft_config:
        return base_model
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("model.peft_config requires the optional dependency `peft` to be installed.") from exc

    peft_raw = dict(model_args.peft_config)
    peft_type = peft_raw.get("type")
    peft_params: Dict[str, Any] = dict(peft_raw.get("params") or {})
    if peft_type != "LoraConfig":
        raise ValueError(f"Unsupported model.peft_config.type: {peft_type!r}. Only 'LoraConfig' is supported.")

    # Keep this wrapper thin: YAML params map directly to PEFT's native
    # LoraConfig, so new PEFT options can be tried without adding local fields.
    peft_config = LoraConfig(**peft_params)
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model


class EmbeddingModel(nn.Module):
    """Encoder wrapper implementing task-aware contrastive embedding loss."""
    def __init__(
        self,
        base_model: PreTrainedModel,
        model_args: ModelConfig,
        training_extras: EmbedTrainingExtras,
    ) -> None:
        super().__init__()
        self.model = base_model
        self.model_args = model_args
        self.training_extras = training_extras
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        if training_extras.negatives_cross_device:
            if not dist.is_available() or not dist.is_initialized():
                logger.info("Distributed is not initialized; cross-device negatives are disabled.")
                self.negatives_cross_device = False
                self.process_rank = 0
                self.world_size = 1
            else:
                self.negatives_cross_device = True
                self.process_rank = dist.get_rank()
                self.world_size = dist.get_world_size()
        else:
            self.negatives_cross_device = False
            self.process_rank = 0
            self.world_size = 1

    def gradient_checkpointing_enable(self, **kwargs):
        return self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        if hasattr(self.model, "enable_input_require_grads"):
            return self.model.enable_input_require_grads(**kwargs)
        return None

    def _pool(self, hidden: Tensor, attention_mask: Tensor, mode: str) -> Tensor:
        if mode == "cls":
            return hidden[:, 0]
        if mode == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if mode == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return hidden[:, -1]
            lengths = attention_mask.sum(dim=1) - 1
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        raise ValueError(f"Unsupported pooling mode: {mode}")

    def _encode_tensor_dict(self, features: Dict[str, Tensor], model_kwargs: Dict[str, object]) -> Tensor:
        outputs = self.model(**features, return_dict=True)
        pooling = str(model_kwargs.get("pooling", self.model_args.pooling))
        reps = self._pool(outputs.last_hidden_state, features["attention_mask"], pooling)
        normalize = bool(model_kwargs.get("normalize_embeddings", self.model_args.normalize_embeddings))
        if normalize:
            reps = F.normalize(reps, dim=-1)
        return reps.contiguous()

    def encode(
        self,
        features: Union[Dict[str, Tensor], List[Dict[str, Tensor]]],
        model_kwargs: Optional[Dict[str, object]] = None,
        sub_batch_size: int = 0,
    ) -> Tensor:
        model_kwargs = model_kwargs or {}
        if isinstance(features, list):
            return torch.cat([self._encode_tensor_dict(f, model_kwargs) for f in features], dim=0).contiguous()
        if sub_batch_size and sub_batch_size > 0:
            reps = []
            size = features["attention_mask"].shape[0]
            for start in range(0, size, sub_batch_size):
                part = {k: v[start : start + sub_batch_size] for k, v in features.items()}
                reps.append(self._encode_tensor_dict(part, model_kwargs))
            return torch.cat(reps, dim=0).contiguous()
        return self._encode_tensor_dict(features, model_kwargs)

    @staticmethod
    def compute_score(q_reps: Tensor, p_reps: Tensor, temperature: float) -> Tensor:
        return torch.matmul(q_reps, p_reps.transpose(0, 1)) / temperature

    @staticmethod
    def _local_scores(q_reps: Tensor, p_reps: Tensor, all_scores: Tensor) -> Tensor:
        """Select each query's contiguous explicit passage group from a score matrix."""
        group_size = p_reps.size(0) // q_reps.size(0)
        base = torch.arange(q_reps.size(0), device=q_reps.device) * group_size
        cols = [all_scores[torch.arange(q_reps.size(0), device=q_reps.device), base + i] for i in range(group_size)]
        return torch.stack(cols, dim=1)

    def _dist_gather_tensor(self, tensor: Tensor) -> Tensor:
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        # all_gather itself is not autograd-aware, so we replace this rank's
        # gathered slot with the original tensor to keep local gradients.
        # Other ranks' reps are negatives only and do not receive gradients here.
        dist.all_gather(gathered, tensor)
        gathered[self.process_rank] = tensor
        return torch.cat(gathered, dim=0).contiguous()

    @staticmethod
    def distill_loss(teacher_targets: Tensor, student_scores: Tensor) -> Tensor:
        return -torch.mean(torch.sum(F.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1))

    def _contrastive_loss(
        self,
        q_reps: Tensor,
        p_reps: Tensor,
        teacher_scores: Optional[Tensor],
        no_in_batch_neg: bool,
        temperature: float,
    ) -> tuple[Tensor, Tensor]:
        """Compute explicit-group or in-batch contrastive loss.

        ``p_reps`` is flattened as one fixed-size group per query, with the
        positive at offset zero. Therefore positive targets in a full score
        matrix are ``query_index * group_size``.
        """
        group_size = p_reps.size(0) // q_reps.size(0)
        teacher_targets = None
        if teacher_scores is not None:
            teacher_targets = F.softmax(teacher_scores.to(q_reps.device).view(q_reps.size(0), -1), dim=-1).detach()

        if no_in_batch_neg:
            # Used by clustering and two-way classification: other examples in
            # the batch may be valid semantic matches, so score only the hard
            # negatives explicitly attached to this query.
            scores = self.compute_score(q_reps, p_reps, temperature)
            local_scores = self._local_scores(q_reps, p_reps, scores)
            targets = torch.zeros(q_reps.size(0), device=q_reps.device, dtype=torch.long)
            loss = self.cross_entropy(local_scores, targets)
            if teacher_targets is not None:
                loss = loss + self.distill_loss(teacher_targets, local_scores)
            return local_scores, loss

        if self.negatives_cross_device and self.training:
            # All ranks have an identical dataset/group-size plan. Gathered
            # passage groups therefore retain the same positive-at-group-start
            # layout used by the target calculation below.
            q_for_score = self._dist_gather_tensor(q_reps)
            p_for_score = self._dist_gather_tensor(p_reps)
            scores = self.compute_score(q_for_score, p_for_score, temperature)
            targets = torch.arange(q_for_score.size(0), device=q_for_score.device, dtype=torch.long) * group_size
            loss = self.cross_entropy(scores, targets)
            if teacher_targets is not None:
                local_scores = self._local_scores(q_for_score, p_for_score, scores)
                local_scores = local_scores[q_reps.size(0) * self.process_rank : q_reps.size(0) * (self.process_rank + 1)]
                loss = loss + self.distill_loss(teacher_targets, local_scores)
            return scores, loss

        scores = self.compute_score(q_reps, p_reps, temperature)
        targets = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long) * group_size
        loss = self.cross_entropy(scores, targets)
        if teacher_targets is not None:
            local_scores = self._local_scores(q_reps, p_reps, scores)
            loss = loss + self.distill_loss(teacher_targets, local_scores)
        return scores, loss

    def forward(
        self,
        queries: Optional[Dict[str, Tensor]] = None,
        passages: Optional[Dict[str, Tensor]] = None,
        teacher_scores: Optional[Tensor] = None,
        dataset_name: Optional[str] = None,
        no_in_batch_neg: bool = False,
        loss_kwargs: Optional[Dict[str, object]] = None,
        model_kwargs: Optional[Dict[str, object]] = None,
        sub_batch_size: int = 0,
        train_group_size: Optional[int] = None,
    ) -> EmbeddingOutput:
        del dataset_name, train_group_size
        if queries is None or passages is None:
            raise ValueError("Both queries and passages are required for training.")
        loss_kwargs = loss_kwargs or {}
        temperature = float(loss_kwargs.get("temperature", self.training_extras.temperature))
        q_reps = self.encode(queries, model_kwargs=model_kwargs, sub_batch_size=sub_batch_size)
        p_reps = self.encode(passages, model_kwargs=model_kwargs, sub_batch_size=sub_batch_size)
        scores, loss = self._contrastive_loss(q_reps, p_reps, teacher_scores, no_in_batch_neg, temperature)
        return EmbeddingOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

    def save(self, output_dir: str) -> None:
        if not self.model_args.peft_config and not self.model_args.peft_adapter_name_or_path:
            logger.info("Saving full model checkpoint to %s", output_dir)
            self.model.save_pretrained(output_dir)
            return

        # PeftModel.save_pretrained saves the adapter files, not a merged base
        # checkpoint. This is the default path because adapter-only saves are
        # small and match PEFT resume/loading behavior.
        logger.info("Saving PEFT adapter checkpoint to %s", output_dir)
        self.model.save_pretrained(output_dir)
        if not self.model_args.save_merged_lora_model:
            return
        if not hasattr(self.model, "merge_and_unload"):
            logger.warning("Current model does not support merge_and_unload; skip merged PEFT save.")
            return
        merged = self.model.merge_and_unload()
        merged_dir = os.path.join(output_dir, "merged")
        logger.info("Saving merged PEFT checkpoint to %s", merged_dir)
        merged.save_pretrained(merged_dir)
