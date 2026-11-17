#!/usr/bin/env python
"""Self-check for the night test-time changes (gamma lookup table on the 127 / 287 crops, anchored template update).

(1) Unit check (always run; no data, no weights, CPU only):
    - the 8-bit gamma 0.6 lookup table equals round(255 * (v / 255) ** 0.6) for all 256 values;
    - SiameseTracker.get_subwindow(..., enhance=f) returns f(crop) of the plain crop (the hook acts on the uint8 crop before the float conversion);
    - HiFTTemplateTracker(mode='feature', k=3, alpha=0.2) with a stub model: after every frame model.zf equals 0.8 * f0 + 0.2 * mean(last 3 feature sets)
      for each of the 3 feature tensors (f0 = the frame-1 features; the last 3 sets include f0 until it drops out).
(2) End-to-end check (optional; needs a CUDA GPU, the first.pth checkpoint and the NAT2021 data):
    python tools/selfcheck_night.py --checkpoint <first.pth> --data_root <dir that contains NAT2021/> --expected tests/expected_night_auc.csv
    runs gamma 0.6 and gamma 0.6 + anchor 0.2 on the sequences listed in the csv and compares the success AUC with the expected values (tolerance 0.001).
    Data layout: <data_root>/NAT2021/anno/<seq>.txt (x,y,w,h per frame, NaN for an absent target) and <data_root>/NAT2021/data_seq/<seq>/*.jpg.
Exit status 0 if every executed check passed.
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)

TOL = 0.001
THRESHOLDS = np.arange(0, 1.05, 0.05)


# ------------------------------------------------------------------ standalone success AUC (same definition as the toolkit's success_overlap)
def success_auc(gt, pred):
    """gt, pred: (n, 4) arrays x, y, w, h. IoU is -1 where the GT has no valid size (w or h not > 0; NaN fails the test); success at each of the 21 thresholds
    0, 0.05, ..., 1 is the share of ALL frames with IoU above the threshold; the AUC is the mean over the thresholds."""
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    iou = np.ones(len(gt)) * -1
    m = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    g, p = gt[m], pred[m]
    iw = np.maximum(0, np.minimum(g[:, 0] + g[:, 2], p[:, 0] + p[:, 2]) - np.maximum(g[:, 0], p[:, 0]))
    ih = np.maximum(0, np.minimum(g[:, 1] + g[:, 3], p[:, 1] + p[:, 3]) - np.maximum(g[:, 1], p[:, 1]))
    inter = iw * ih
    iou[m] = np.clip(inter / (g[:, 2] * g[:, 3] + p[:, 2] * p[:, 3] - inter), 0, 1)
    return float(np.mean([(iou > t).sum() / float(len(gt)) for t in THRESHOLDS]))


def load_gt(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in re.split(r'[,\s]+', line)[:4]])
    return np.array(rows)


# ------------------------------------------------------------------ (1) unit checks
def unit_checks():
    import torch
    from pysot.core.config import cfg
    cfg.merge_from_file(os.path.join(ROOT, 'experiments', 'config.yaml'))
    cfg.CUDA = False
    from pysot.utils.enhance import make_enhancer
    from pysot.tracker.hift_tracker import HiFTTracker
    from pysot.tracker.hift_tracker_tmpl import HiFTTemplateTracker
    ok = True

    # gamma lookup table on all 256 values
    enh = make_enhancer('gamma', 0.6, 3.0, 'none')
    patch = np.stack([np.arange(256, dtype=np.uint8).reshape(16, 16)] * 3, axis=-1)
    expected = (255.0 * (np.arange(256) / 255.0) ** 0.6).round().astype(np.uint8).reshape(16, 16)
    out = enh(np.ascontiguousarray(patch))
    good = bool((out[..., 0] == expected).all() and (out[..., 1] == expected).all() and (out[..., 2] == expected).all())
    print('[unit] gamma 0.6 lookup table equals the expected curve on all 256 values: {}'.format('PASS' if good else 'FAIL'))
    ok &= good

    class Stub(object):
        """no weights: template() draws deterministic random 'features'; track() returns random score / box maps"""
        def __init__(self):
            self.zf = None
            self.history = []
            self.gen = torch.Generator().manual_seed(0)

        def eval(self):
            return self

        def template(self, z):
            self.zf = tuple(torch.randn(1, c, 3, 3, generator=self.gen) for c in (4, 5, 6))
            self.history.append(self.zf)

        def track(self, x):
            n = cfg.TRAIN.OUTPUT_SIZE
            return {'loc': torch.rand(1, 4, n, n, generator=self.gen) * 0.2 - 0.1,
                    'cls1': torch.randn(1, 2, n, n, generator=self.gen),
                    'cls2': torch.randn(1, 1, n, n, generator=self.gen)}

    # hook: get_subwindow(..., enhance=f) == f(plain crop)
    tr = HiFTTracker(Stub())
    rng = np.random.RandomState(0)
    img = rng.randint(0, 256, (240, 320, 3)).astype(np.uint8)
    avg = img.mean(axis=(0, 1))
    plain = tr.get_subwindow(img, np.array([150., 110.]), 127, 90, avg, None).numpy()[0].transpose(1, 2, 0).astype(np.uint8)
    hooked = tr.get_subwindow(img, np.array([150., 110.]), 127, 90, avg, enh).numpy()[0].transpose(1, 2, 0).astype(np.uint8)
    good = bool(np.array_equal(hooked, enh(np.ascontiguousarray(plain))))
    print('[unit] get_subwindow(enhance=gamma) equals gamma applied to the plain uint8 crop: {}'.format('PASS' if good else 'FAIL'))
    ok &= good

    # anchored template update through the real tracker class
    stub = Stub()
    tr = HiFTTemplateTracker(stub, mode='feature', k=3, alpha=0.2)
    frames = [rng.randint(0, 256, (240, 320, 3)).astype(np.uint8) for _ in range(7)]
    tr.init(frames[0], [140., 100., 40., 30.])
    good = True
    for n, f in enumerate(frames[1:], start=1):
        tr.track(f)
        sets = stub.history[:n + 1]                       # f0, f1, ..., fn as produced by the stub
        last3 = sets[-3:]
        for i in range(3):
            want = 0.8 * sets[0][i] + 0.2 * torch.stack([s[i] for s in last3]).mean(0)
            if not torch.allclose(stub.zf[i], want, atol=1e-6):
                good = False
    print('[unit] template after each of 6 frames == 0.8 * f0 + 0.2 * mean(last 3 feature sets), all 3 tensors: {}'.format('PASS' if good else 'FAIL'))
    ok &= good
    return ok


# ------------------------------------------------------------------ (2) end-to-end
def run_sequence(model, make_tracker, data_root, seq):
    import cv2
    from pysot.utils.bbox import get_axis_aligned_bbox
    ddir = os.path.join(data_root, 'NAT2021', 'data_seq', seq)
    names = sorted(n for n in os.listdir(ddir) if n.lower().endswith(('.jpg', '.jpeg', '.png')))
    gt = load_gt(os.path.join(data_root, 'NAT2021', 'anno', seq + '.txt'))
    assert len(gt) == len(names), (seq, len(gt), len(names))
    tracker = make_tracker()
    boxes = []
    for i, n in enumerate(names):
        img = cv2.imread(os.path.join(ddir, n))
        if i == 0:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt[0]))
            box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            tracker.init(img, box)
            boxes.append(box)
        else:
            boxes.append(tracker.track(img)['bbox'])
    return success_auc(gt, np.array(boxes)), len(names)


def end_to_end(args):
    import torch
    from pysot.core.config import cfg
    cfg.merge_from_file(os.path.join(ROOT, 'experiments', 'config.yaml'))
    cfg.CUDA = True                      # the unit check switches it off; the end-to-end run uses the GPU
    if not torch.cuda.is_available():
        print('[e2e] SKIPPED: no CUDA GPU available (the model class builds on the GPU)')
        return True
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.hift_tracker import HiFTTracker
    from pysot.tracker.hift_tracker_tmpl import HiFTTemplateTracker
    from pysot.utils.model_load import load_pretrain
    cfg.TRACK.ENH_TEMPLATE = cfg.TRACK.ENH_SEARCH = 'gamma'
    cfg.TRACK.ENH_GAMMA = 0.6
    cfg.TRACK.ENH_CLIP = 3.0
    cfg.TRACK.ENH_DENOISE = 'none'
    torch.set_num_threads(1)
    model = load_pretrain(ModelBuilder(), args.checkpoint).cuda().eval()
    rows = list(csv.DictReader(open(args.expected)))
    seqs = sorted({r['sequence'] for r in rows})
    exp = {(r['sequence'], r['model']): float(r['success_auc']) for r in rows}
    makers = {'gamma': lambda: HiFTTracker(model),                      # plain tracker with gamma 0.6 on both crops
              'gamma_anchor': lambda: HiFTTemplateTracker(model, mode='feature', k=3, alpha=0.2)}
    ok = True
    print('[e2e] sequence  model         frames  AUC(this run)  AUC(expected)  |diff|   seconds')
    for s in seqs:
        for mname in ('gamma', 'gamma_anchor'):
            t0 = time.time()
            auc, n = run_sequence(model, makers[mname], args.data_root, s)
            d = abs(auc - exp[(s, mname)])
            good = d <= TOL
            ok &= good
            print('[e2e] {:8s} {:13s} {:6d}  {:13.6f}  {:13.6f}  {:.6f}  {:7.1f}  {}'.format(s, mname, n, auc, exp[(s, mname)], d, time.time() - t0, 'PASS' if good else 'FAIL'))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--data_root', default='')
    ap.add_argument('--expected', default=os.path.join(ROOT, 'tests', 'expected_night_auc.csv'))
    args = ap.parse_args()
    ok = unit_checks()
    if args.checkpoint and args.data_root:
        ok &= end_to_end(args)
    else:
        print('[e2e] not run (give --checkpoint and --data_root to run it)')
    print('SELFCHECK', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
