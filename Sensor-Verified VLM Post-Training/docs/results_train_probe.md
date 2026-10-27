| Model | Method | n | Total | Hands | Object | Point |
|---|---|---|---|---|---|---|
| - | best constant answer | | 0.358 | | | |
| Qwen3-VL-4B | SFT | 240 | **0.805** | 0.846 | 0.787 | 0.678 |
| Qwen3-VL-4B | SFT + GRPO | 240 | **0.802** | 0.836 | 0.796 | 0.674 |
| Qwen3-VL-8B | SFT | 240 | **0.810** | 0.846 | 0.821 | 0.669 |
| Qwen3-VL-8B | SFT + GRPO | 240 | **0.802** | 0.842 | 0.812 | 0.649 |

Change from SFT on the same train_probe samples (mean, 95% paired bootstrap interval):

| Model | Method | Total | Hands | Point |
|---|---|---|---|---|
| Qwen3-VL-4B | SFT + GRPO | -0.003 [-0.013, +0.008] | -0.010 [-0.020, +0.000] | -0.004 [-0.029, +0.025] |
| Qwen3-VL-8B | SFT + GRPO | -0.009 [-0.023, +0.005] | -0.004 [-0.018, +0.010] | -0.021 [-0.054, +0.013] |

Fraction of frames in which a hand is said to be busy:

| | Left | Right |
|---|---|---|
| labels | 0.54 | 0.67 |
| Qwen3-VL-4B SFT | 0.58 | 0.74 |
| Qwen3-VL-4B SFT + GRPO | 0.61 | 0.77 |
| Qwen3-VL-8B SFT | 0.58 | 0.73 |
| Qwen3-VL-8B SFT + GRPO | 0.63 | 0.74 |
