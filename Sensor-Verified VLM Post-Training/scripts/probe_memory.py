"""Hardware feasibility probe: can one SFT / GRPO / DPO step of a 4-bit
Qwen3-VL + LoRA fit on this GPU?

Each phase runs one training step on a synthetic 2x2 frame tile and reports
peak allocated memory and wall time. OOM is caught per phase and reported
instead of crashing, so a single run maps out what fits.

    python scripts/probe_memory.py --model Qwen/Qwen3-VL-2B-Instruct --compute fp32

The vision tower is frozen, so image features are computed once and fed back
through `mm_encoder_outputs`, as the training pipeline will do with cached
features. Non-quantized weights are stored in fp16: this GPU (Turing, sm_75)
has no native bf16, and neither does a Kaggle T4. `--offload` keeps the
vision tower and the (tied) embedding / LM head on the CPU in fp32.
"""

import argparse
import gc
import time

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

PROMPT = (
    "The image tiles 4 frames (1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right) "
    "from an egocentric video. In which frame does a hand first grasp an object, "
    "which hand is it, and where is the contact point in that frame? "
    'Answer as JSON: {"frame": int, "hand": "left|right", "point": [x, y]}.'
)
ANSWER = '{"frame": 3, "hand": "right", "point": [0.62, 0.41]}'
REJECTED = '{"frame": 2, "hand": "left", "point": [0.10, 0.90]}'
CPU_MODULES = ["model.visual", "model.language_model.embed_tokens", "lm_head"]


def make_tile(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    half = size // 2
    colors = [(200, 80, 80), (80, 200, 80), (80, 80, 200), (200, 200, 80)]
    for i, color in enumerate(colors):
        x, y = (i % 2) * half, (i // 2) * half
        draw.rectangle([x, y, x + half - 1, y + half - 1], fill=color)
        draw.ellipse([x + half // 3, y + half // 3, x + half // 2, y + half // 2], fill=(240, 220, 200))
    return img


def gpu_mb(x: int) -> str:
    return f"{x / 2**20:,.0f} MB"


def load(model_id: str, dtype: torch.dtype, lora_rank: int, offload: bool, grad_ckpt: bool):
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "quantization_config"):
        # pre-quantized checkpoint: transformers ignores a passed config's offload flag
        config.quantization_config["llm_int8_enable_fp32_cpu_offload"] = offload
        quant = None
    else:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["lm_head", "visual"],
            llm_int8_enable_fp32_cpu_offload=offload,
        )
    device_map = {**{m: "cpu" for m in CPU_MODULES}, "": 0} if offload else {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, config=config, quantization_config=quant, dtype=torch.float16, device_map=device_map
    )
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            m.compute_dtype = dtype
    if offload:
        # CPU kernels for fp16 matmuls are slow; fp32 costs only host RAM
        model.model.visual.float()
        model.model.language_model.embed_tokens.float()
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,
        # language model only; the vision tower stays frozen
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def encode_prompt(model, processor, image):
    """Tokenized prompt (on GPU) and its image features, computed once."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    pixel_values = prompt.pop("pixel_values")
    visual = model.get_base_model().model.visual
    dev = next(visual.parameters()).device
    t0 = time.perf_counter()
    with torch.no_grad():
        feats = model.get_base_model().model.get_image_features(pixel_values.to(dev), prompt["image_grid_thw"].to(dev))
    feats.pooler_output = tuple(f.to(0) for f in feats.pooler_output)
    feats.deepstack_features = [tuple(f.to(0) for f in layer) for layer in feats.deepstack_features]
    print(f"image features: {feats.pooler_output[0].shape[0]} tokens, encoded once in {time.perf_counter() - t0:.1f}s on {dev}")
    return {k: v.to(0) for k, v in prompt.items()}, feats


def repeat_feats(feats, b: int):
    return type(feats)(
        pooler_output=feats.pooler_output * b,
        deepstack_features=[layer * b for layer in feats.deepstack_features],
    )


def build_inputs(processor, prompt, feats, answers: list[str]):
    """Prompt + each answer, batched; returns inputs and the (padded) answer length."""
    tok = processor.tokenizer
    answer_ids = [tok(a + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids for a in answers]
    n_ans = max(a.shape[1] for a in answer_ids)
    answer_ids = torch.cat([F.pad(a, (0, n_ans - a.shape[1]), value=tok.pad_token_id) for a in answer_ids]).to(0)
    b = len(answers)
    seq_len = prompt["input_ids"].shape[1]
    inputs = {"mm_encoder_outputs": {"image": repeat_feats(feats, b)}}
    for k, v in prompt.items():
        if k == "input_ids":
            inputs[k] = torch.cat([v.repeat(b, 1), answer_ids], dim=1)
        elif v.dim() == 2 and v.shape == (1, seq_len):
            # per-token tensors (attention_mask, mm_token_type_ids): extend over the answer
            fill = 1 if k == "attention_mask" else 0
            inputs[k] = torch.cat([v.repeat(b, 1), torch.full((b, n_ans), fill, dtype=v.dtype, device=v.device)], dim=1)
        elif k == "image_grid_thw":
            inputs[k] = v.repeat(b, 1)
        else:
            inputs[k] = v
    return inputs, n_ans


def answer_logprobs(model, inputs, n_ans):
    """Per-token log-probs of the last n_ans tokens; only those logits are materialized."""
    out = model(**inputs, logits_to_keep=n_ans + 1)
    logits = out.logits[:, :-1].float()
    targets = inputs["input_ids"][:, -n_ans:].to(logits.device)
    return torch.gather(logits.log_softmax(-1), 2, targets.unsqueeze(-1)).squeeze(-1).to(0)


def phase(name, fn, params):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        info = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[{name}] OK   peak {gpu_mb(torch.cuda.max_memory_allocated())}  "
              f"reserved {gpu_mb(torch.cuda.max_memory_reserved())}  {dt:.1f}s  {info or ''}")
    except torch.OutOfMemoryError:
        print(f"[{name}] OOM  peak {gpu_mb(torch.cuda.max_memory_allocated())}")
    finally:
        for p in params:
            p.grad = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--compute", choices=["fp16", "fp32"], default="fp32", help="4-bit matmul compute dtype")
    ap.add_argument("--offload", action="store_true", help="vision tower + embeddings/LM head on CPU")
    ap.add_argument("--tile", type=int, default=448, help="tile side in pixels (32 px per visual token)")
    ap.add_argument("--group", type=int, default=4, help="GRPO samples per prompt")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--grpo-micro", type=int, default=1, help="GRPO samples per forward/backward micro-batch")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.compute]

    torch.manual_seed(0)
    t0 = time.perf_counter()
    model = load(args.model, dtype, args.lora_rank, args.offload, not args.no_grad_ckpt)
    processor = AutoProcessor.from_pretrained(args.model)
    model_params_with_grad = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(model_params_with_grad, lr=1e-5)
    n_lora = sum(p.numel() for p in model_params_with_grad)
    print(f"{args.model}  compute={args.compute}  offload={args.offload}  grad_ckpt={not args.no_grad_ckpt}  "
          f"grpo_micro={args.grpo_micro}  load {time.perf_counter() - t0:.0f}s  "
          f"GPU weights {gpu_mb(torch.cuda.memory_allocated())}  lora params {n_lora / 1e6:.1f}M")

    prompt, feats = encode_prompt(model, processor, make_tile(args.tile))

    def sft():
        model.train()
        inputs, n_ans = build_inputs(processor, prompt, feats, [ANSWER])
        loss = -answer_logprobs(model, inputs, n_ans).mean()
        loss.backward()
        opt.step()
        return f"seq {inputs['input_ids'].shape[1]} tok  loss {loss.item():.3f}"

    def grpo():
        model.eval()
        t_gen = time.perf_counter()
        with torch.no_grad():
            # generate() expands the features in place for num_return_sequences: pass a copy
            seqs = model.generate(**prompt, mm_encoder_outputs={"image": repeat_feats(feats, 1)}, do_sample=True, temperature=1.0,
                                  num_return_sequences=args.group, max_new_tokens=args.max_new, use_cache=True)
        t_gen = time.perf_counter() - t_gen
        completions = processor.tokenizer.batch_decode(seqs[:, prompt["input_ids"].shape[1]:], skip_special_tokens=True)
        # gradient accumulation over micro-batches of completions
        model.train()
        adv = torch.randn(args.group).to(0)
        tok = processor.tokenizer
        for i in range(0, args.group, args.grpo_micro):
            inputs, n_ans = build_inputs(processor, prompt, feats, completions[i:i + args.grpo_micro])
            mask = (inputs["input_ids"][:, -n_ans:] != tok.pad_token_id).float()
            logp = answer_logprobs(model, inputs, n_ans)
            per_seq = (logp * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            loss = -(adv[i:i + args.grpo_micro] * per_seq).sum() / args.group
            loss.backward()
        opt.step()
        ok = sum(torch.isfinite(p.grad).all().item() for p in model_params_with_grad if p.grad is not None)
        return f"gen {t_gen:.1f}s for {args.group}x{args.max_new} tok  finite grads {ok}/{len(model_params_with_grad)}  sample: {completions[0][:60]!r}"

    def dpo():
        model.train()
        inputs, n_ans = build_inputs(processor, prompt, feats, [ANSWER, REJECTED])
        mask = (inputs["input_ids"][:, -n_ans:] != processor.tokenizer.pad_token_id).float()
        with torch.no_grad(), model.disable_adapter():
            ref = (answer_logprobs(model, inputs, n_ans) * mask.to(0)).sum(-1)
        pol = (answer_logprobs(model, inputs, n_ans) * mask.to(0)).sum(-1)
        margin = (pol[0] - ref[0]) - (pol[1] - ref[1])
        loss = -F.logsigmoid(0.1 * margin)
        loss.backward()
        opt.step()
        return f"loss {loss.item():.4f}"

    for name, fn in [("SFT ", sft), ("GRPO", grpo), ("DPO ", dpo)]:
        for _ in range(args.repeat):
            phase(name, fn, model_params_with_grad)


if __name__ == "__main__":
    main()
