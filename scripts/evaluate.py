"""Score YOLO predictions against KITTI ground truth (IoU 0.5).

--mapping v2 (default): cyclist unions, TruckVan + Tram groups.
--mapping v1: session-2 mapping, for comparison.
--mapping v3 (session 7): grouping identical to v2, but Pedestrian
ground truth claims its person boxes BEFORE unions form, so a genuine
pedestrian standing near a bike can no longer be consumed by a union.
Slices are per group: occlusion, truncation, height band.

`evaluate()` is importable so scripts/sweep_iou.py and
scripts/sweep_conf.py can loop thresholds without re-running the
detector."""
import argparse
import json
from pathlib import Path

from mapping import MAPS, build_gt_groups, build_pred_groups
from metrics import average_precision, calibration_bins, match_image

ROOT = Path(__file__).resolve().parent.parent


def height_band(bbox):
    h = bbox[3] - bbox[1]
    return "<25" if h < 25 else ("25-40" if h <= 40 else "40+")


def evaluate(gt, preds, mapping, iou_threshold):
    """Score predictions against ground truth. Returns the metrics dict
    that used to be written straight to metrics.json (same schema)."""
    _, _, groups = MAPS[mapping]
    stats = {g: {"tp": 0, "fp": 0, "fn": 0, "scores": [], "tp_flags": []} for g in groups}
    slices = {g: {"occlusion": {}, "truncation": {}, "height": {}} for g in groups}
    ignored_gt = {}
    ignored_pred = {}
    for iid, gt_objs in gt.items():
        gt_by_group, ig = build_gt_groups(gt_objs, mapping)
        image_preds = preds.get(iid, [])
        ped_matched = None
        claimed = set()
        person_preds = []
        if mapping == "v3":
            # Pedestrian-first: Pedestrian GT claims its person boxes
            # before unions form. Claimed persons leave the pool, so
            # unions form only from leftovers.
            person_preds = [p for p in image_preds if p["class"] == "person"]
            _, ped_matched = match_image(gt_by_group["Pedestrian"], person_preds, iou_threshold)
            claimed = set(ped_matched.keys())
            remaining = []
            person_i = 0
            for p in image_preds:
                if p["class"] == "person":
                    if person_i not in claimed:
                        remaining.append(p)
                    person_i += 1
                else:
                    remaining.append(p)
            pred_by_group, ip = build_pred_groups(remaining, mapping)
        else:
            pred_by_group, ip = build_pred_groups(image_preds, mapping)
        for o in ig:
            ignored_gt[o["type"]] = ignored_gt.get(o["type"], 0) + 1
        for p in ip:
            ignored_pred[p["class"]] = ignored_pred.get(p["class"], 0) + 1
        for g in groups:
            if mapping == "v3" and g == "Pedestrian":
                # Pedestrian stats come from the first-pass match:
                # claimed persons are TPs; leftover non-union persons
                # are FPs (greedy matching guarantees no leftover
                # person matches a free Pedestrian GT).
                s = stats[g]
                s["tp"] += len(claimed)
                s["fp"] += len(pred_by_group[g])
                matched_gt = set(ped_matched.values())
                s["fn"] += len(gt_by_group[g]) - len(matched_gt)
                for i, p in enumerate(person_preds):
                    if i in claimed:
                        s["scores"].append(p["score"])
                        s["tp_flags"].append(True)
                for p in pred_by_group[g]:
                    s["scores"].append(p["score"])
                    s["tp_flags"].append(False)
                for j, o in enumerate(gt_by_group[g]):
                    missed = j not in matched_gt
                    for key, val in (("occlusion", str(o["occluded"])),
                                     ("truncation", "none" if o["truncated"] == 0 else "partial"),
                                     ("height", height_band(o["bbox"]))):
                        d = slices[g][key].setdefault(val, {"n": 0, "missed": 0})
                        d["n"] += 1
                        d["missed"] += int(missed)
                continue
            tp_flags, matched = match_image(gt_by_group[g], pred_by_group[g], iou_threshold)
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
                    d = slices[g][key].setdefault(val, {"n": 0, "missed": 0})
                    d["n"] += 1
                    d["missed"] += int(missed)
    out_groups = {}
    for g in groups:
        s = stats[g]
        paired = sorted(zip(s["scores"], s["tp_flags"]), key=lambda x: -x[0])
        ap_value = average_precision([f for _, f in paired], s["tp"] + s["fn"])
        out_groups[g] = {
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0,
            "recall": s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0,
            "ap": ap_value, "n_gt": s["tp"] + s["fn"],
        }
    mAP = sum(v["ap"] for v in out_groups.values()) / len(groups)
    out_slices = {}
    for g in groups:
        out_slices[g] = {}
        for key, vals in slices[g].items():
            out_slices[g][key] = {k: {"n": v["n"], "missed": v["missed"],
                                       "miss_rate": v["missed"] / v["n"] if v["n"] else 0.0}
                                  for k, v in sorted(vals.items())}
    all_scores = [sc for g in groups for sc in stats[g]["scores"]]
    all_flags = [f for g in groups for f in stats[g]["tp_flags"]]
    return {
        "mapping": mapping,
        "iou_threshold": iou_threshold,
        "groups": out_groups,
        "mAP": mAP,
        "slices": out_slices,
        "calibration": {
            "overall": calibration_bins(all_flags, all_scores),
            "by_group": {g: calibration_bins(stats[g]["tp_flags"], stats[g]["scores"]) for g in groups},
        },
        "ignored": {"ground_truth_by_type": ignored_gt, "predictions_by_class": ignored_pred},
    }


def print_report(result):
    """Print the per-group table + worst slices, as evaluate.py always did."""
    groups = list(result["groups"])
    print(f"Mapping: {result['mapping']}, IoU threshold: {result['iou_threshold']}")
    print(f"{'Group':<12}{'TP':>6}{'FP':>6}{'FN':>6}{'Prec':>7}{'Rec':>7}{'AP':>7}")
    for g in groups:
        v = result["groups"][g]
        print(f"{g:<12}{v['tp']:>6}{v['fp']:>6}{v['fn']:>6}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['ap']:>7.3f}")
    print(f"mAP: {result['mAP']:.3f}")
    flat = [(f"{g}/{key}={val}", d["miss_rate"], d["n"])
            for g in groups for key, vals in result["slices"][g].items() for val, d in vals.items() if d["n"] >= 20]
    print("Worst slices by miss rate (n>=20):")
    for name, rate, n in sorted(flat, key=lambda x: -x[1])[:8]:
        print(f"  {name}: miss rate {rate:.3f} (n={n})")
    print(f"Ignored GT: {result['ignored']['ground_truth_by_type']}")
    print(f"Ignored predictions: {result['ignored']['predictions_by_class']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="evals/ground_truth.json")
    ap.add_argument("--pred", default="evals/predictions.json")
    ap.add_argument("--output", default="evals/metrics.json")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--mapping", choices=["v1", "v2", "v3"], default="v2")
    args = ap.parse_args()
    gt = json.loads((ROOT / args.gt).read_text())
    preds = json.loads((ROOT / args.pred).read_text())
    result = evaluate(gt, preds, args.mapping, args.iou)
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print_report(result)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
