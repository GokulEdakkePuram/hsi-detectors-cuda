#!/usr/bin/env python3
"""Score a dumped SAM angle map against HyperBlood ground truth.

    build/src/hsi_detect --source=envi \
      --envi=data/hyperblood_prepared/F_1.hdr \
      --library=data/hyperblood_targets.csv \
      --frames=1 --warmup=0 --dump-score=/tmp/F_1.f32

    python3 scripts/eval_detection.py /tmp/F_1.f32 \
      data/hyperblood_prepared/F_1_gt.u8 --width=696 --height=520

Reports ROC AUC, the operating point at a chosen threshold, and - the number
that actually matters here - how far blood separates from each of its visual
lookalikes. A detector that reaches 0.99 AUC against background while scoring
ketchup as close to blood is not useful, and an aggregate figure hides that.

Depends on numpy only. AUC is computed from the rank sum rather than by
integrating a curve, which is exact and needs no extra package.
"""

import argparse
import pathlib
import sys

import numpy as np

CLASS_NAMES = {
    0: "background",
    1: "blood",
    2: "ketchup",
    3: "artificial_blood",
    4: "beetroot_juice",
    5: "poster_paint",
    6: "tomato_concentrate",
    7: "acrylic_paint",
    8: "uncertain_blood",
}


def roc_auc(scores, labels):
    """AUC via the Mann-Whitney statistic, with ties given average rank.

    Scores here are spectral angles, where smaller means a better match, so
    they are negated to keep the usual "higher score is more positive"
    convention.
    """
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # Average the ranks within each run of equal scores, otherwise ties are
    # broken by array order and the AUC drifts with the input's layout.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, scores.size + 1):
        if end == scores.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    rank_sum = ranks[labels == 1].sum()
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("angle_map", type=pathlib.Path, help="raw float32 from --dump-score")
    parser.add_argument("ground_truth", type=pathlib.Path, help="raw uint8 class mask")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--target", type=int, default=1, help="positive class [1=blood]")
    parser.add_argument("--threshold", type=float, default=0.10, help="radians")
    args = parser.parse_args()

    pixels = args.width * args.height
    angles = np.fromfile(args.angle_map, dtype="<f4")
    truth = np.fromfile(args.ground_truth, dtype=np.uint8)
    for name, array in (("angle map", angles), ("ground truth", truth)):
        if array.size != pixels:
            raise SystemExit(
                f"{name} has {array.size} entries, expected {pixels} "
                f"({args.height}x{args.width})"
            )

    # Unannotated and bookkeeping labels are excluded rather than counted as
    # negatives: calling an unlabelled pixel a false positive would be a guess
    # dressed up as a measurement.
    valid = truth <= 8
    labels = (truth[valid] == args.target).astype(np.int8)
    scores = -angles[valid].astype(np.float64)

    if labels.sum() == 0:
        raise SystemExit(
            f"class {args.target} ({CLASS_NAMES.get(args.target)}) is absent from "
            f"{args.ground_truth.name}"
        )

    auc = roc_auc(scores, labels)
    predicted = angles[valid] <= args.threshold
    actual = labels == 1
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")

    name = CLASS_NAMES.get(args.target, str(args.target))
    print(f"target class      {args.target} ({name})")
    print(f"annotated pixels  {int(valid.sum())} of {pixels}")
    print(f"positives         {int(labels.sum())}")
    print(f"\nROC AUC           {auc:.4f}")
    print(f"\nat threshold {args.threshold:.3f} rad")
    print(f"  precision       {precision:.4f}")
    print(f"  recall          {recall:.4f}")
    print(f"  f1              {f1:.4f}")
    print(f"  tp {tp}  fp {fp}  fn {fn}  tn {tn}")

    # The real question: does the angle separate blood from the things that
    # look like blood? A high AUC against background is easy and says little.
    print("\nmean spectral angle by class (lower is a closer match)")
    target_mean = float(angles[valid][actual].mean())
    for label in sorted(np.unique(truth[valid])):
        selected = angles[valid][truth[valid] == label]
        if selected.size == 0:
            continue
        gap = float(selected.mean()) - target_mean
        marker = "  <- target" if label == args.target else f"  gap {gap:+.4f}"
        print(
            f"  {CLASS_NAMES.get(int(label), str(label)):>20} "
            f"n={selected.size:7d}  mean {selected.mean():.4f}  "
            f"min {selected.min():.4f}{marker}"
        )

    if np.isnan(auc):
        sys.exit(1)


if __name__ == "__main__":
    main()
