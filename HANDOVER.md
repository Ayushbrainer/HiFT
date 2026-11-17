# HANDOVER: night test-time changes (gamma on both crops, anchored template)

Commit on this branch: the 7 changed files of the cherry-picked commit (message "night test-time changes: gamma and anchored template"), plus hand-over files (this file, `docs/NIGHT_SUMMARY.md`, `tools/selfcheck_night.py`, `tests/expected_night_auc.csv`, `night_changes.patch` = `git diff <base> <that commit>`). Base = upstream HiFT as in the repository HEAD before these changes. The full experiment history is on branch `night-full`.

## What changed (7 files, all optional at the defaults: with no flags the tracker behaves as before)
| File | What it does |
|---|---|
| `pysot/utils/enhance.py` (new) | `make_enhancer(mode, gamma, clip, denoise)` returns `f(uint8 BGR patch) -> uint8 BGR patch` or `None`. Mode `gamma` = 8-bit lookup table `out = round(255 * (in / 255) ** gamma)` applied with `cv2.LUT`. (`clahe`, `gamma_clahe` and a denoiser option exist for experiments; they are not part of the recommended setting.) |
| `pysot/tracker/base_tracker.py` | `SiameseTracker.get_subwindow(..., enhance=None)`: after the crop is cut and resized to 127 (template) or 287 (search) the optional `enhance` is applied to the uint8 patch, before the float32 conversion. |
| `pysot/tracker/hift_tracker.py` | builds `enh_t` and `enh_s` from `cfg.TRACK.ENH_TEMPLATE / ENH_SEARCH / ENH_GAMMA / ENH_CLIP / ENH_DENOISE` and passes them to the two `get_subwindow` calls; adds `reset_box()` (diagnostic, unused by default). |
| `pysot/core/config.py` | adds the five `TRACK.ENH_*` options (defaults `none`, `none`, 0.4, 3.0, `none`, so nothing changes unless a flag is given). |
| `pysot/tracker/hift_tracker_tmpl.py` (new) | `HiFTTemplateTracker(model, mode='feature', k=3, alpha=0.2)`: the anchored template update (below). |
| `tools/test.py` | adds `--enh_template --enh_search --denoise --gamma --clip --tag --every --video_list` (and a GT-based diagnostic `--reinit_every`, off by default); plain tracker. |
| `tools/test_tmpl.py` (new) | same driver with the template tracker: `--tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2`. |

## The settings
- **Gamma 0.6** applied to BOTH the 127 x 127 template crop and the 287 x 287 search crop, as an 8-bit lookup table on the uint8 BGR crop before the network (the network input stays 0-255 float, no other normalisation).
- **Anchored template** (after every tracked frame): `template features = 0.8 * (frame-1 template features) + 0.2 * mean(features of the last 3 template crops)`, applied separately to each of the 3 backbone output tensors (`model.zf` is a tuple of 3 tensors). The new template crop is cut at the new box (same context rule, same gamma). The last-3 window starts with the frame-1 features, so the first two updates still include them.

## Flags and examples (run from the repository root; `--gamma` defaults to 0.4, so always pass 0.6)
(a) gamma only
```
python tools/test.py --dataset <DATASET> --snapshot <path/to/first.pth> --config experiments/config.yaml \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tag _gamma06
```
(b) gamma + anchor
```
python tools/test_tmpl.py --dataset <DATASET> --snapshot <path/to/first.pth> --config experiments/config.yaml \
    --enh_template gamma --enh_search gamma --gamma 0.6 --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 --tag _gamma06_anchor02
```
Results go to `tools/results/<DATASET>/<snapshot name + w1>/<tag>`. `<DATASET>` must be one the toolkit knows; the NAT / UAVDark135 dataset adapter is not part of this branch (it is commit "night evaluation" on `night-full`). The standalone self-check below needs no adapter.

## What the anchor requires from a model class
- `model.template(z)`: runs the backbone on a 127 x 127 crop `z` (1 x 3 x 127 x 127 float32, 0-255) and stores the result in `model.zf`;
- `model.zf` is a tuple / list of 3 tensors (it is blended and averaged element-wise, tensor by tensor);
- the tracker reads `model.zf` when it calls `model.track(x)`.
So any model that keeps its template in `zf` (or an attribute you rename in `hift_tracker_tmpl.py`) and can compute it from a 127 px crop works.

## Cost
- Gamma: one `cv2.LUT` per crop (256-entry table), negligible.
- Anchor: ONE EXTRA BACKBONE PASS PER FRAME on the 127 x 127 crop (plus a small blend of 3 tensors). Nothing else changes.

## Self-check
```
python tools/selfcheck_night.py                       # unit check: no data, no weights, CPU, a few seconds
python tools/selfcheck_night.py --checkpoint <first.pth> --data_root <dir containing NAT2021/> --expected tests/expected_night_auc.csv   # end-to-end, needs a CUDA GPU
```
Unit check: gamma 0.6 table on all 256 values; `get_subwindow` hook; template update `0.8 f0 + 0.2 mean(last 3)` on all 3 tensors through the real tracker class with a stub model. End-to-end: gamma and gamma + anchor on 3 NAT2021 sequences (N02006, N08019, N04005; the first has 52 absent-target frames) against `tests/expected_night_auc.csv` within 0.001 success AUC, using a small standalone success-AUC function (same definition as the pysot toolkit's `success_overlap`: IoU -1 where the GT has no valid size, all frames in the denominator).

## Notes for CPU-efficient models
1. **Gamma only is recommended.** It is an 8-bit lookup table on the crops before the network, so it does not depend on the network.
2. **The anchor costs one extra backbone pass per frame on the 127 px crop.** Its CPU speed cost has not been measured by us. Measure it before use.
3. **Before claiming any gain**, run `none`, `gamma` and `gamma + anchor` on UAVDark135 test with the efficient model and report success AUC and time per tracking step.
4. **Where to hook gamma in the efficient pipeline (the crop step):**
```python
import cv2, numpy as np
GAMMA_LUT = (255.0 * (np.arange(256) / 255.0) ** 0.6).round().astype(np.uint8)

def crop_for_network(img, centre, original_size, model_size, channel_mean):
    patch = cut_and_pad(img, centre, original_size, channel_mean)      # your existing crop (uint8 BGR)
    patch = cv2.resize(patch, (model_size, model_size))                # 127 for the template, 287 for the search crop
    patch = cv2.LUT(patch, GAMMA_LUT)                                   # <- the only new line; apply to BOTH crops
    return to_network_input(patch)                                      # float32 / tensor conversion as before
```
5. **The effect on any other model must be tested.** In our ablation, run A (fine-tuned without gamma augmentation) was hurt by gamma 0.6: success AUC on UAVDark135 test fell from 0.468 to 0.426 and on the held-out daytime set from 0.607 to 0.595 (logbook entry 71 in `docs/experiments_log.md` on `night-full`; also added to `night_evidence/claims.md`). A model trained without such augmentation can lose with gamma.
