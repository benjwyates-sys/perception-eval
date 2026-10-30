"""Diagnose cyclist misses: for each Cyclist ground truth, are the person
and bicycle components even present?

Buckets per cyclist GT (raw saved predictions, conf >= 0.25 only — a
component detected below that floor counts as absent):
  both / person_only / bike_only / neither
where 'present' means a box of that class overlaps the GT box (IoU > 0,
the same permissiveness as the union rule). Each bucket is cross-tabbed
with whether that GT matched under mapping v2, plus height-band and
occlusion breakdowns.

Reading: misses in neither/one-component buckets are detector blindness
(no matching fix recovers them); misses in both-present-but-unmatched
are the residual matching problem."""
import json
from pathlib import Path

from evaluate import height_band
from mapping import build_gt_groups, build_pred_groups
from metrics import iou, match_image

ROOT = Path(__file__).resolve().parent.parent


def main():
    gt = json.loads((ROOT / "evals" / "ground_truth.json").read_text())
    preds = json.loads((ROOT / "evals" / "predictions.json").read_text())
    buckets = {b: {"n": 0, "matched": 0} for b in ("both", "person_only", "bike_only", "neither")}
    by_height = {}
    by_occlusion = {}
    for iid, gt_objs in gt.items():
        gt_by_group, _ = build_gt_groups(gt_objs, "v2")
        pred_by_group, _ = build_pred_groups(preds.get(iid, []), "v2")
        _, matched = match_image(gt_by_group["Cyclist"], pred_by_group["Cyclist"], 0.5)
        matched_gt = set(matched.values())
        image_preds = preds.get(iid, [])
        persons = [p for p in image_preds if p["class"] == "person"]
        bicycles = [p for p in image_preds if p["class"] == "bicycle"]
        for j, o in enumerate(gt_by_group["Cyclist"]):
            has_p = any(iou(o["bbox"], p["bbox"]) > 0 for p in persons)
            has_b = any(iou(o["bbox"], b["bbox"]) > 0 for b in bicycles)
            bucket = "both" if has_p and has_b else ("person_only" if has_p else ("bike_only" if has_b else "neither"))
            m = j in matched_gt
            buckets[bucket]["n"] += 1
            buckets[bucket]["matched"] += int(m)
            for table, key in ((by_height, height_band(o["bbox"])), (by_occlusion, str(o["occluded"]))):
                d = table.setdefault(bucket, {}).setdefault(key, {"n": 0, "matched": 0})
                d["n"] += 1
                d["matched"] += int(m)
    out = {"buckets": buckets, "by_height": by_height, "by_occlusion": by_occlusion}
    out_path = ROOT / "evals" / "cyclist_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("Cyclist component diagnostic (mapping v2 matching, IoU 0.5):")
    print(f"{'Bucket':<12}{'n':>5}{'matched':>9}{'missed':>8}")
    for b in ("both", "person_only", "bike_only", "neither"):
        v = buckets[b]
        print(f"{b:<12}{v['n']:>5}{v['matched']:>9}{v['n'] - v['matched']:>8}")
    print("By height band (n / matched):")
    for b in ("both", "person_only", "bike_only", "neither"):
        print(f"  {b}: " + ", ".join(f"{k} {v['n']}/{v['matched']}" for k, v in sorted(by_height.get(b, {}).items())))
    print("By occlusion (n / matched):")
    for b in ("both", "person_only", "bike_only", "neither"):
        print(f"  {b}: " + ", ".join(f"{k} {v['n']}/{v['matched']}" for k, v in sorted(by_occlusion.get(b, {}).items())))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
