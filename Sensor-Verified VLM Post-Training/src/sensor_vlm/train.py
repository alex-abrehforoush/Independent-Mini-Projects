"""Shared training pieces for SFT, DPO and GRPO.

Batches are left-padded so that every completion ends at the last position;
only the logits of the final `n_keep` positions are materialized (the vocabulary
is 152k, so full-sequence logits alone would not fit on a 4 GB GPU). A
per-sample mask marks which of those positions belong to its completion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image

from sensor_vlm.model import load_model, messages_for

END = "<|im_end|>"


def answer_text(sample: dict) -> str:
    return json.dumps(sample["answer"])


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        # language model only; the vision tower stays frozen
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def trainable_model(model_id: str, adapter: str | None = None, lora_r: int = 16, fp16: bool = False):
    """4-bit base + a trainable LoRA: a fresh one, or `adapter` (e.g. the SFT result) to keep training.

    fp16=True runs the vision tower and 4-bit matmuls in fp16, for GPUs with fast fp16 (T4);
    pair it with loss scaling (`torch.amp.GradScaler`).
    """
    model, processor = load_model(model_id, vision_fp32=not fp16, compute_fp32=not fp16)
    if adapter is None:
        return add_lora(model, r=lora_r, alpha=2 * lora_r), processor
    from peft import PeftModel

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model, processor


def encode_prompt(processor, sample: dict, tasks_dir: Path) -> dict:
    image = Image.open(tasks_dir / sample["image"]).convert("RGB")
    return processor.apply_chat_template(
        messages_for(sample, image), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )


def collate(processor, prompts: list[dict], completions: list[str]) -> tuple[dict, torch.Tensor]:
    """Left-padded batch of prompt + completion (+ <|im_end|>), and the (B, n_keep) completion mask."""
    tok = processor.tokenizer
    comp = [tok(c + END, add_special_tokens=False, return_tensors="pt").input_ids[0] for c in completions]
    seqs = [torch.cat([p["input_ids"][0], c]) for p, c in zip(prompts, comp)]
    L, n_keep = max(len(s) for s in seqs), max(len(c) for c in comp)
    B = len(seqs)

    input_ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    mm_types = torch.zeros((B, L), dtype=torch.long)
    for i, (p, s) in enumerate(zip(prompts, seqs)):
        input_ids[i, L - len(s):] = s
        attention_mask[i, L - len(s):] = 1
        if "mm_token_type_ids" in p:
            mm_types[i, L - len(s):L - len(s) + p["mm_token_type_ids"].shape[1]] = p["mm_token_type_ids"][0]
    mask = torch.zeros((B, n_keep), dtype=torch.float32)
    for i, c in enumerate(comp):
        mask[i, n_keep - len(c):] = 1

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": torch.cat([p["pixel_values"] for p in prompts]),
        "image_grid_thw": torch.cat([p["image_grid_thw"] for p in prompts]),
    }
    if "mm_token_type_ids" in prompts[0]:
        batch["mm_token_type_ids"] = mm_types
    return batch, mask


def completion_logprobs(model, batch: dict, n_keep: int) -> torch.Tensor:
    """Log-prob of each of the last n_keep tokens given its prefix: (B, n_keep), fp32."""
    batch = {k: v.to(0) for k, v in batch.items()}
    out = model(**batch, logits_to_keep=n_keep + 1)
    logits = out.logits[:, :-1].float()
    targets = batch["input_ids"][:, -n_keep:]
    return torch.gather(logits.log_softmax(-1), 2, targets.unsqueeze(-1)).squeeze(-1)


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class JsonlLog:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "a")

    def truncate_after(self, step: int) -> None:
        """Drop rows logged after `step` (work lost past the checkpoint a run resumes from)."""
        self.f.close()
        rows = [line for line in self.path.read_text().splitlines() if json.loads(line).get("step", 0) <= step]
        self.path.write_text("".join(r + "\n" for r in rows))
        self.f = open(self.path, "a")

    def __call__(self, **row):
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
