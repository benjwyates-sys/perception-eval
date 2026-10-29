"""Sweep the confidence threshold; find each group's operating point.

Unlike the IoU sweep (session 5, a scoring knob), the confidence
threshold is a DEPLOYMENT knob: it decides which boxes survive at all.
Raising it trades precision up against recall down, so F1 should peak
somewhere in the middle — a real interior optimum, not the edge answer
session 5 produced.

Fast: no detector rerun. The saved predictions already contain
everything above YOLO's 0.25 default, so thresholds at or above 0.25
are pure filtering. Going below 0.25 would need a fresh predict run
with a lower conf and is not covered here.

Caveat for the writeup: AP at each confidence threshold is a
truncated-list AP (the ranked list just ends earlier), so
precision/recall/F1 are the primary numbers and AP is secondary.
Scoring is IoU 0.5, mapping v2 throughout."""
import json
from pathlib import Path

from evaluate import evaluate

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    gt = json.loads((ROOT / "evals" / "ground_truth.json").read_text())
    preds = json.loads((ROOT / "evals" / "predictions.json").read_text())
    rows = {}
    for t in THRESHOLDS:
        filtered = {iid: [p for p in plist if p["score"] >= t] for iid, plist in preds.items()}
        res = evaluate(gt, filtered, "v2", 0.5)
        rows[str(t)] = {
            "mAP": res["mAP"],
            "n_predictions": sum(len(v) for v in filtered.values()),
            "groups": {g: {"precision": v["precision"], "recall": v["recall"],
                           "ap": v["ap"], "f1": f1(v["precision"], v["recall"]),
                           "n_pred": v["tp"] + v["fp"]}
                      for g, v in res["groups"].items()},
        }
    out_path = ROOT / "evals" / "conf_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2))
    groups = list(rows["0.5"]["groups"])
    print("F1 by confidence threshold (IoU 0.5, mapping v2):")
    print(f"{'Conf':>5}" + "".join(f"{g + ' F1':>14}" for g in groups) + f"{'mAP':>8}{'n_pred':>8}")
    for t in THRESHOLDS:
        r = rows[str(t)]
        print(f"{t:>5}" + "".join(f"{r['groups'][g]['f1']:>14.3f}" for g in groups) + f"{r['mAP']:>8.3f}{r['n_predictions']:>8}")
    print("F1-optimal confidence threshold per group:")
    for g in groups:
        best = max(THRESHOLDS, key=lambda t: rows[str(t)]["groups"][g]["f1"])
        print(f"  {g}: {best} (F1 {rows[str(best)]['groups'][g]['f1']:.3f})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
