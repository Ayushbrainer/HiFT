# Night tracking with HiFT: what we changed (one page)

## What we changed
Two test-time changes to the unchanged first.pth tracker; no retraining:
1. **Gamma 0.6 on both crops.** The 127 px template crop and the 287 px search crop are brightened with the same 8-bit lookup table before the network.
2. **Anchored template.** The template features are 0.8 x the frame-1 template features + 0.2 x the mean of the features of the last 3 frames' template crops (for each of the 3 backbone tensors).

## Intuition (a motivation, not a tested mechanism)
Night crops are dark. Brightening both crops identically makes them look more like the footage the network was trained on. The first-frame template is the only template known to be correct, so we keep 80% of it and let 20% follow the recent appearance.

## Headline (mean success AUC, first.pth)
| Benchmark (videos) | original | gamma 0.6 | gamma 0.6 + anchor 0.2 |
|---|---|---|---|
| NAT2021 (180) | 0.372 | 0.417 | 0.453 |
| NAT2021L (23) | 0.299 | 0.316 | 0.377 |
| UAVDark135 test (23) | 0.352 | 0.434 | 0.443 |

Paired bootstrap, 95% interval over videos (difference in mean success AUC):
| Benchmark | gamma - original | anchor + gamma - original | anchor + gamma - gamma |
|---|---|---|---|
| NAT2021 | +0.0449 [+0.0215, +0.0680] | +0.0812 [+0.0562, +0.1076] | +0.0363 [+0.0175, +0.0554] |
| NAT2021L | +0.0172 [-0.0224, +0.0563] | +0.0781 [+0.0092, +0.1537] | +0.0609 [-0.0086, +0.1280] |
| UAVDark135 test | +0.0826 [+0.0323, +0.1445] | +0.0916 [+0.0028, +0.1739] | +0.0090 [-0.0545, +0.0631] |

On the held-out UAV123 daytime set (30 videos) none of the three differences is distinguishable from 0.

## Experiments in one line each ("lost" = predicted box centre more than 20 px from the ground-truth centre)
- **Effect of each setting with intervals:** the table above.
- **How the tracker loses the target (NAT2021, anchor + gamma):** 95.0% of lost-stretch starts (864 of 909) have the target centre inside the search window; 86.3% of lost frames (43,793 of 50,749) are in stretches of 101 frames or more.
- **Template similarity:** the template cropped at the true box has a mean cosine similarity to the frame-1 template of 0.467 on lost frames and 0.597 on found frames (ROC AUC 0.737).
- **Runner-up peaks:** a logged second peak is within 20 px of the target on 53.7% of the first lost frames, falling to 27.5% at the 5th lost frame, 12.2% at the 20th and 4.7% over lost frames of stretches of 101+ frames.
- **Fine-tuning on night data (runs A-D):** with anchor + gamma all four selected checkpoints are within 0.013 of first.pth on NAT2021 (A -0.012, B -0.004, C -0.009, D -0.013); on other sets some runs are further away.
- **Oracle template bank (templates cut from ground-truth boxes, an upper bound, not a tracker):** NAT2021 0.489 against 0.417 for the same code with one template; on NAT2021L 0.370 against 0.316.

## Caveats
- Gamma 0.6 and anchor 0.2 were tuned on NAT2021 videos, and NAT2021 is also a headline benchmark, so those numbers are selection-biased.
- The independent check on UAVDark135 test supports gamma more than the anchor (anchor on top of gamma: +0.0090, interval includes 0).
- NAT2021L is not independent of NAT2021 (13 of its 23 videos start from the same first box as a NAT2021 video).
- Single runs; the 23-video benchmarks have wide intervals.
- The analysis scripts are our own, not the official toolkit; only the per-video success and precision numbers come from the official pysot evaluation code.

Code, job scripts and the logbook (docs/experiments_log.md) are on branch night-full (commit eb54372e43b39731192950ac1ddc1c26fa77ab10). The per-video CSVs, claims.md and figures are in the separate night_evidence package, not in git.
