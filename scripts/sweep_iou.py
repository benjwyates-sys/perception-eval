"""Sweep the IoU matching threshold; report per-group metrics + calibration.

The IoU threshold is an EVALUATION knob, not a deployment knob: it sets
how strictly a prediction must localize to count as a hit. Lower = lenient
(more TPs, higher recall); higher = strict. The sweep answers two questions:
  1. Sensitivity: do the session 3/4 conclusions survive stricter/looser
     scoring, or do they depend on the 0.5 bar? (Cyclist unions are the
     prime suspect: union boxes are approximate, so cyclist AP should be
     the most IoU-sensitive group.)
  2. Optimization: at which threshold is each group's precision/recall
     balance best, by F1? Reported as information, not as a deployment
     choice — the deployment dial is the confidence threshold (session 6).

Fast: no detector rerun, just re-matching the saved predictions.json
against ground_truth.json at each threshold."""
import json
from pathlib import Path

from evaluate import evaluate

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    gt = json.loads((ROOT / "evals" / "ground_truth.json").read_text())
    preds = json.loads((ROOT / "evals" / "predictions.json").read_text())
    rows = {}
    for t in THRESHOLDS:
        res = evaluate(gt, preds, "v2", t)
        rows[str(t)] = {
            "mAP": res["mAP"],
            "groups": {g: {"precision": v["precision"], "recall": v["recall"],
                           "ap": v["ap"], "f1": f1(v["precision"], v["recall"])}
                      for g, v in res["groups"].items()},
            "calibration_overall": res["calibration"]["overall"],
        }
    out_path = ROOT / "evals" / "iou_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2))
    groups = list(rows["0.5"]["groups"])
    print("AP by IoU threshold (mapping v2):")
    print(f"{'IoU':>5}" + "".join(f"{g + ' AP':>14}" for g in groups) + f"{'mAP':>8}")
    for t in THRESHOLDS:
        r = rows[str(t)]
        print(f"{t:>5}" + "".join(f"{r['groups'][g]['ap']:>14.3f}" for g in groups) + f"{r['mAP']:>8.3f}")
    print("F1-optimal IoU threshold per group:")
    for g in groups:
        best = max(THRESHOLDS, key=lambda t: rows[str(t)]["groups"][g]["f1"])
        print(f"  {g}: {best} (F1 {rows[str(best)]['groups'][g]['f1']:.3f})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
