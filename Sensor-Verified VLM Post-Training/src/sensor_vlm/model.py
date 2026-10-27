"""Loading Qwen3-VL for training and evaluation on small GPUs.

Language-model linear layers are 4-bit NF4 (bitsandbytes); everything else is
stored in fp16 because Turing GPUs (GTX 1650, Kaggle T4) have no bf16. On a
GTX 1650 fp16 math is very slow (the vision tower takes 9.4 s per 448 px tile
in fp16 vs 0.40 s in fp32), so the vision tower can run in fp32 there; a T4
has fast fp16 and does not need it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def load_model(model_id: str, adapter: str | None = None, vision_fp32: bool = True, compute_fp32: bool = True):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32 if compute_fp32 else torch.float16,
        llm_int8_skip_modules=["lm_head", "visual"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0}
    )
    if vision_fp32:
        model.model.visual.float()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def messages_for(sample: dict, image: Image.Image) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": sample["prompt"]}]}]


def load_split(tasks_dir: Path, split: str, limit: int | None = None) -> list[dict]:
    with open(tasks_dir / f"{split}.jsonl") as f:
        samples = [json.loads(line) for line in f]
    return samples[:limit] if limit else samples


@torch.no_grad()
def generate(model, processor, samples: list[dict], tasks_dir: Path, max_new_tokens: int = 96,
             do_sample: bool = False, temperature: float = 1.0, num_return_sequences: int = 1) -> list[str]:
    """Completions for a batch of samples, num_return_sequences per sample, sample-major order."""
    msgs = [messages_for(s, Image.open(tasks_dir / s["image"]).convert("RGB")) for s in samples]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": True},
    ).to(0)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                         temperature=temperature if do_sample else None, top_p=None, top_k=None,
                         num_return_sequences=num_return_sequences)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
