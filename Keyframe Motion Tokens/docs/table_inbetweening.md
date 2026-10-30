| method | L2Q 5 | L2Q 15 | L2Q 30 | L2Q 45 | L2P 5 | L2P 15 | L2P 30 | L2P 45 | NPSS 5 | NPSS 15 | NPSS 30 | NPSS 45 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SLERP (official script, this repo) | 0.22 | 0.62 | 0.98 | 1.25 | 0.37 | 1.25 | 2.32 | 3.45 | 0.0023 | 0.0391 | 0.2013 | 0.4493 |
| RMIB (published) | 0.17 | 0.42 | 0.69 | 0.94 | 0.23 | 0.65 | 1.28 | 2.24 | 0.0020 | 0.0260 | 0.1330 | 0.3300 |
| ∆-I (published) | 0.11 | 0.32 | 0.57 | – | 0.13 | 0.47 | 1.00 | – | 0.0014 | 0.0220 | 0.1220 | – |
| TST (published) | 0.10 | 0.28 | 0.54 | 0.87 | 0.10 | 0.39 | 0.89 | 1.68 | 0.0011 | 0.0190 | 0.1120 | 0.3200 |
| SILK (published) | 0.11 | 0.27 | 0.50 | 0.79 | 0.13 | 0.38 | 0.83 | 1.59 | 0.0012 | 0.0180 | 0.1050 | 0.3000 |
| **keyframe tokens, FSQ** | 0.53 | 0.74 | 0.95 | 1.16 | 0.75 | 1.27 | 1.95 | 2.72 | 0.0078 | 0.0474 | 0.1762 | 0.4069 |
| **keyframe tokens, continuous + flow** | 0.48 | 0.79 | 1.07 | 1.30 | 0.65 | 1.49 | 2.57 | 3.68 | 0.0067 | 0.0534 | 0.2154 | 0.4961 |
| **fixed-rate tokens, FSQ** | 0.21 | 0.48 | 0.77 | 1.05 | 0.31 | 0.88 | 1.51 | 2.27 | 0.0023 | 0.0303 | 0.1557 | 0.3832 |
| **fixed-rate tokens, continuous + flow** | 0.21 | 0.45 | 0.75 | 1.06 | 0.29 | 0.75 | 1.44 | 2.23 | 0.0023 | 0.0289 | 0.1545 | 0.3861 |
| **keyframe tokens, FSQ, uniform keys** | 0.49 | 0.67 | 0.91 | 1.13 | 0.70 | 1.09 | 1.78 | 2.55 | 0.0091 | 0.0473 | 0.1742 | 0.3904 |
| **keyframe tokens, continuous raw key features + flow (ablation)** | 0.47 | 0.85 | 1.20 | 1.47 | 0.90 | 2.04 | 3.61 | 5.21 | 0.0059 | 0.0600 | 0.2744 | 0.6068 |

| method | gap | L2P | foot skate, own contacts cm/s (GT) | foot skate, GT contacts cm/s (GT) | jitter / GT | penetration cm | diversity APD cm | tokens or keys |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLERP | 15 | 1.25 | 91.9 (7.9) | 64.9 (3.7) | 0.60 | 0.58 | 0 | 0.0 |
| keyframe tokens, FSQ | 15 | 1.27 | 45.3 (7.9) | 44.8 (3.7) | 2.78 | 0.20 | 7.3 | 3.9 |
| keyframe tokens, continuous + flow | 15 | 1.49 | 43.4 (7.9) | 50.6 (3.7) | 2.87 | 0.14 | 8.7 | 3.9 |
| fixed-rate tokens, FSQ | 15 | 0.88 | 42.6 (7.9) | 30.7 (3.7) | 0.93 | 0.24 | 2.9 | 5.0 |
| fixed-rate tokens, continuous + flow | 15 | 0.75 | 34.7 (7.9) | 27.0 (3.7) | 0.87 | 0.17 | 2.7 | 5.0 |
| keyframe tokens, FSQ, uniform keys | 15 | 1.09 | 41.7 (7.9) | 38.2 (3.7) | 1.45 | 0.18 | 5.4 | 3.0 |
| SLERP | 45 | 3.45 | 87.0 (8.1) | 87.6 (3.8) | 0.26 | 0.61 | 0 | 0.0 |
| keyframe tokens, FSQ | 45 | 2.72 | 39.8 (8.1) | 60.6 (3.8) | 1.58 | 0.24 | 18.2 | 11.0 |
| keyframe tokens, continuous + flow | 45 | 3.68 | 50.0 (8.1) | 85.3 (3.8) | 2.19 | 0.29 | 22.6 | 11.1 |
| fixed-rate tokens, FSQ | 45 | 2.27 | 55.9 (8.1) | 53.4 (3.8) | 0.66 | 0.55 | 10.7 | 12.0 |
| fixed-rate tokens, continuous + flow | 45 | 2.23 | 51.9 (8.1) | 55.3 (3.8) | 0.64 | 0.44 | 10.1 | 12.0 |
| keyframe tokens, FSQ, uniform keys | 45 | 2.55 | 38.4 (8.1) | 54.7 (3.8) | 1.24 | 0.27 | 14.4 | 11.0 |
| SLERP | 90 | 7.39 | 68.2 (7.5) | 71.1 (3.8) | 0.15 | 0.63 | 0 | 0.0 |
| keyframe tokens, FSQ | 90 | 5.68 | 33.4 (7.5) | 73.3 (3.8) | 1.10 | 0.24 | 35.3 | 20.7 |
| keyframe tokens, continuous + flow | 90 | 7.45 | 41.0 (7.5) | 97.2 (3.8) | 1.68 | 0.31 | 45.3 | 21.3 |
| fixed-rate tokens, FSQ | 90 | 5.45 | 56.8 (7.5) | 73.0 (3.8) | 0.54 | 0.73 | 24.2 | 23.0 |
| fixed-rate tokens, continuous + flow | 90 | 5.26 | 58.3 (7.5) | 75.8 (3.8) | 0.53 | 0.65 | 26.6 | 23.0 |
| keyframe tokens, FSQ, uniform keys | 90 | 5.56 | 30.9 (7.5) | 69.7 (3.8) | 0.83 | 0.27 | 29.6 | 22.0 |

| tokens | code head | L2P 5 | L2P 15 | L2P 45 | L2P 90 | foot skate cm/s @45 |
|---|---|---:|---:|---:|---:|---:|
| fixed-rate tokens, FSQ | joint softmax per group | 0.31 | 0.91 | 2.35 | 5.55 | 57.1 |
| fixed-rate tokens, FSQ | softmax per dimension | 0.31 | 0.88 | 2.27 | 5.45 | 53.4 |
| keyframe tokens, FSQ | joint softmax per group | 0.97 | 2.08 | 4.59 | 8.97 | 96.6 |
| keyframe tokens, FSQ | softmax per dimension | 0.75 | 1.27 | 2.72 | 5.68 | 60.6 |

| tokens | gap | emitted bits/s | generator cross-entropy bits/s | one-level code error: cm moved | frames moved > 1 cm | ground-truth codes cm | generator (greedy) cm | SLERP cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| keyframe tokens, FSQ | 5 | 361 | 218 (60%) | 0.76 | 27% | 3.05 | 4.61 | 2.10 |
| fixed-rate tokens, FSQ | 5 | 478 | 184 (38%) | 0.20 | 0% | 1.80 | 1.88 | 2.10 |
| keyframe tokens, FSQ, uniform keys | 5 | 239 | 125 (52%) | 1.08 | 48% | 3.08 | 4.44 | 2.10 |
| keyframe tokens, FSQ | 15 | 339 | 231 (68%) | 0.83 | 31% | 3.30 | 8.30 | 7.61 |
| fixed-rate tokens, FSQ | 15 | 399 | 185 (46%) | 0.37 | 3% | 3.73 | 5.62 | 7.61 |
| keyframe tokens, FSQ, uniform keys | 15 | 239 | 160 (67%) | 0.89 | 33% | 3.12 | 7.07 | 7.61 |
| keyframe tokens, FSQ | 45 | 326 | 245 (75%) | 1.39 | 41% | 3.25 | 18.50 | 24.60 |
| fixed-rate tokens, FSQ | 45 | 319 | 173 (54%) | 0.47 | 10% | 6.20 | 14.81 | 24.60 |
| keyframe tokens, FSQ, uniform keys | 45 | 292 | 218 (75%) | 1.35 | 37% | 3.15 | 17.03 | 24.60 |
| keyframe tokens, FSQ | 90 | 328 | 255 (78%) | 2.02 | 41% | 3.28 | 40.84 | 57.80 |
| fixed-rate tokens, FSQ | 90 | 306 | 172 (56%) | 0.50 | 13% | 7.36 | 38.97 | 57.80 |
| keyframe tokens, FSQ, uniform keys | 90 | 292 | 228 (78%) | 2.40 | 41% | 3.14 | 39.48 | 57.80 |
