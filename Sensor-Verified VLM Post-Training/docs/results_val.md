| Model | Method | n | Total | Hands | Object | Point |
|---|---|---|---|---|---|---|
| - | best constant answer | | 0.400 | | | |
| Qwen3-VL-2B | Zero-shot | 332 | **0.432** | 0.527 | 0.355 | 0.119 |
| Qwen3-VL-4B | Zero-shot | 332 | **0.482** | 0.591 | 0.364 | 0.208 |
| Qwen3-VL-4B | SFT | 332 | **0.624** | 0.758 | 0.518 | 0.364 |
| Qwen3-VL-4B | SFT + GRPO | 332 | **0.626** | 0.750 | 0.509 | 0.391 |
| Qwen3-VL-4B | SFT + DPO | 332 | **0.611** | 0.721 | 0.515 | 0.373 |
| Qwen3-VL-8B | Zero-shot | 332 | **0.513** | 0.612 | 0.410 | 0.257 |
| Qwen3-VL-8B | SFT | 332 | **0.638** | 0.775 | 0.539 | 0.367 |
| Qwen3-VL-8B | SFT + GRPO | 332 | **0.629** | 0.760 | 0.536 | 0.358 |

Change from SFT on the same val samples (mean, 95% paired bootstrap interval):

| Model | Method | Total | Hands | Point |
|---|---|---|---|---|
| Qwen3-VL-4B | SFT + GRPO | +0.002 [-0.012, +0.014] | -0.008 [-0.018, +0.003] | +0.028 [+0.000, +0.055] |
| Qwen3-VL-4B | SFT + DPO | -0.014 [-0.029, +0.001] | -0.037 [-0.054, -0.021] | +0.009 [-0.024, +0.043] |
| Qwen3-VL-8B | SFT + GRPO | -0.009 [-0.023, +0.005] | -0.015 [-0.029, -0.003] | -0.009 [-0.046, +0.028] |

Fraction of frames in which a hand is said to be busy:

| | Left | Right |
|---|---|---|
| labels | 0.37 | 0.66 |
| Qwen3-VL-2B Zero-shot | 0.77 | 0.76 |
| Qwen3-VL-4B Zero-shot | 0.53 | 0.68 |
| Qwen3-VL-4B SFT | 0.54 | 0.76 |
| Qwen3-VL-4B SFT + GRPO | 0.58 | 0.78 |
| Qwen3-VL-4B SFT + DPO | 0.66 | 0.84 |
| Qwen3-VL-8B Zero-shot | 0.67 | 0.74 |
| Qwen3-VL-8B SFT | 0.51 | 0.70 |
| Qwen3-VL-8B SFT + GRPO | 0.59 | 0.72 |
