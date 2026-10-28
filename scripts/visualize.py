"""Draw GT + prediction boxes on eval images; build an HTML gallery.

Colors (BGR): green = ground truth, blue = true positive, red = false
positive, gray = unevaluated ground truth, yellow = ignored prediction.
Uses cv2, which arrives with ultralytics — no new dependency."""
import argparse
import json
from pathlib import Path

import cv2

from mapping import MAPS, build_gt_groups, build_pred_groups
from metrics import match_image

ROOT = Path(__file__).resolve().parent.parent


def draw(img, bbox, color, label):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="comma-separated image ids")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--mapping", choices=["v1", "v2"], default="v2")
    args = ap.parse_args()
    split = json.loads((ROOT / "evals" / "eval_split.json").read_text())
    gt = json.loads((ROOT / "evals" / "ground_truth.json").read_text())
    preds = json.loads((ROOT / "evals" / "predictions.json").read_text())
    if args.images:
        ids = [s.strip() for s in args.images.split(",")]
    elif args.all:
        ids = split["image_ids"]
    else:
        ids = split["image_ids"][: args.n]
    out_dir = ROOT / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid in ids:
        img = cv2.imread(str(ROOT / "data" / "training" / "image_2" / f"{iid}.png"))
        if img is None:
            print(f"missing image {iid}, skipping")
            continue
        gt_objs = gt.get(iid, [])
        gt_map = MAPS[args.mapping][0]
        groups = MAPS[args.mapping][2]
        gt_by_group, _ = build_gt_groups(gt_objs, args.mapping)
        pred_by_group, ignored_preds = build_pred_groups(preds.get(iid, []), args.mapping)
        for o in gt_objs:
            g = gt_map.get(o["type"])
            draw(img, o["bbox"], (0, 180, 0) if g else (128, 128, 128),
                 f"GT {o['type']}" + ("" if g else " (unevaluated)"))
        tp_total = fp_total = 0
        for g in groups:
            tp_flags, _ = match_image(gt_by_group[g], pred_by_group[g], args.iou)
            for flag, p in zip(tp_flags, pred_by_group[g]):
                draw(img, p["bbox"], (255, 120, 0) if flag else (0, 0, 255),
                     f"{p['class']} {p['score']:.2f}" + (" UNION" if p.get("union") else "") + ("" if flag else " FP"))
                tp_total += int(flag)
                fp_total += int(not flag)
        for p in ignored_preds:
            draw(img, p["bbox"], (0, 200, 200), f"{p['class']} {p['score']:.2f} (ignored)")
        fn = sum(len(v) for v in gt_by_group.values()) - tp_total
        cv2.imwrite(str(out_dir / f"{iid}.png"), img)
        rows.append((iid, len(gt_objs), tp_total, fp_total, fn))
    html = ["<html><head><style>body{font-family:sans-serif}img{max-width:100%}</style></head><body>"]
    html.append("<p>Green = ground truth, blue = true positive, red = false positive, gray = unevaluated ground truth, yellow = ignored prediction.</p>")
    for iid, n_gt, tp, fp, fn in rows:
        html.append(f"<h3>{iid} — GT {n_gt}, TP {tp}, FP {fp}, FN {fn}</h3><img src='{iid}.png'>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))
    print(f"wrote {len(rows)} annotated images + gallery -> {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
