"""Score YOLO predictions against KITTI ground truth (IoU 0.5).

Evaluation classes are KITTI's main benchmark trio: Car, Pedestrian,
Cyclist. COCO->KITTI mapping: car->Car, person->Pedestrian (GT Pedestrian
and Person_sitting grouped — the detector can't tell them apart),
bicycle->Cyclist. Other COCO predictions and Van/Truck/Tram/Misc ground
truth are counted as ignored, not scored."""
import argparse
import json
from pathlib import Path

from metrics import average_precision, calibration_bins, match_image

ROOT = Path(__file__).resolve().parent.parent
CLASS_MAP = {"car": "Car", "person": "Pedestrian", "bicycle": "Cyclist"}
GT_GROUP = {"Car": "Car", "Pedestrian": "Pedestrian", "Person_sitting": "Pedestrian", "Cyclist": "Cyclist"}
GROUPS = ["Car", "Pedestrian", "Cyclist"]


def height_band(bbox):
    h = bbox[3] - bbox[1]
    return "<25" if h < 25 else ("25-40" if h <= 40 else "40+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="evals/ground_truth.json")
    ap.add_argument("--pred", default="evals/predictions.json")
    ap.add_argument("--output", default="evals/metrics.json")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    gt = json.loads((ROOT / args.gt).read_text())
    preds = json.loads((ROOT / args.pred).read_text())
    stats = {g: {"tp": 0, "fp": 0, "fn": 0, "scores": [], "tp_flags": []} for g in GROUPS}
    slices = {"occlusion": {}, "truncation": {}, "height": {}}
    ignored_gt = {}
    ignored_pred = {}
    for iid, gt_objs in gt.items():
        gt_by_group = {g: [] for g in GROUPS}
        for o in gt_objs:
            g = GT_GROUP.get(o["type"])
            if g is None:
                ignored_gt[o["type"]] = ignored_gt.get(o["type"], 0) + 1
            else:
                gt_by_group[g].append(o)
        pred_by_group = {g: [] for g in GROUPS}
        for p in preds.get(iid, []):
            g = CLASS_MAP.get(p["class"])
            if g is None:
                ignored_pred[p["class"]] = ignored_pred.get(p["class"], 0) + 1
            else:
                pred_by_group[g].append(p)
        for g in GROUPS:
            tp_flags, matched = match_image(gt_by_group[g], pred_by_group[g], args.iou)
            s = stats[g]
            s["tp"] += sum(tp_flags)
            s["fp"] += len(tp_flags) - sum(tp_flags)
            matched_gt = set(matched.values())
            s["fn"] += len(gt_by_group[g]) - len(matched_gt)
            for flag, p in zip(tp_flags, pred_by_group[g]):
                s["scores"].append(p["score"])
                s["tp_flags"].append(flag)
            for j, o in enumerate(gt_by_group[g]):
                missed = j not in matched_gt
                for key, val in (("occlusion", str(o["occluded"])),
                                 ("truncation", "none" if o["truncated"] == 0 else "partial"),
                                 ("height", height_band(o["bbox"]))):
                    d = slices[key].setdefault(val, {"n": 0, "missed": 0})
                    d["n"] += 1
                    d["missed"] += int(missed)
    out_groups = {}
    for g in GROUPS:
        s = stats[g]
        paired = sorted(zip(s["scores"], s["tp_flags"]), key=lambda x: -x[0])
        ap_value = average_precision([f for _, f in paired], s["tp"] + s["fn"])
        out_groups[g] = {
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0,
            "recall": s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0,
            "ap": ap_value, "n_gt": s["tp"] + s["fn"],
        }
    mAP = sum(v["ap"] for v in out_groups.values()) / len(GROUPS)
    out_slices = {}
    for key, vals in slices.items():
        out_slices[key] = {k: {"n": v["n"], "missed": v["missed"],
                                "miss_rate": v["missed"] / v["n"] if v["n"] else 0.0}
                           for k, v in sorted(vals.items())}
    all_scores = [sc for g in GROUPS for sc in stats[g]["scores"]]
    all_flags = [f for g in GROUPS for f in stats[g]["tp_flags"]]
    result = {
        "iou_threshold": args.iou,
        "groups": out_groups,
        "mAP": mAP,
        "slices": out_slices,
        "calibration": {
            "overall": calibration_bins(all_flags, all_scores),
            "by_group": {g: calibration_bins(stats[g]["tp_flags"], stats[g]["scores"]) for g in GROUPS},
        },
        "ignored": {"ground_truth_by_type": ignored_gt, "predictions_by_class": ignored_pred},
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"IoU threshold: {args.iou}")
    print(f"{'Group':<12}{'TP':>6}{'FP':>6}{'FN':>6}{'Prec':>7}{'Rec':>7}{'AP':>7}")
    for g in GROUPS:
        v = out_groups[g]
        print(f"{g:<12}{v['tp']:>6}{v['fp']:>6}{v['fn']:>6}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['ap']:>7.3f}")
    print(f"mAP: {mAP:.3f}")
    flat = [(f"{key}={val}", d["miss_rate"], d["n"]) for key, vals in out_slices.items() for val, d in vals.items() if d["n"] >= 20]
    print("Worst slices by miss rate (n>=20):")
    for name, rate, n in sorted(flat, key=lambda x: -x[1])[:5]:
        print(f"  {name}: miss rate {rate:.3f} (n={n})")
    print(f"Ignored GT: {ignored_gt}")
    print(f"Ignored predictions: {ignored_pred}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
