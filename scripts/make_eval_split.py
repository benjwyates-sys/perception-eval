"""Carve a held-out eval split from KITTI's labeled training images."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_label(path):
    objs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] == "DontCare":
            continue
        objs.append({
            "type": parts[0],
            "truncated": float(parts[1]),
            "occluded": int(parts[2]),
            "bbox": [float(x) for x in parts[4:8]],
        })
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="evals/eval_split.json")
    ap.add_argument("--gt-output", default="evals/ground_truth.json")
    args = ap.parse_args()
    image_dir = ROOT / "data" / "training" / "image_2"
    label_dir = ROOT / "data" / "training" / "label_2"
    ids = sorted(p.stem for p in image_dir.glob("*.png"))
    assert len(ids) == 7481, f"expected 7481 images, found {len(ids)} — run fetch_kitti.py first"
    rng = random.Random(args.seed)
    eval_ids = sorted(rng.sample(ids, args.n))
    gt = {}
    counts = {}
    for iid in eval_ids:
        objs = parse_label(label_dir / f"{iid}.txt")
        gt[iid] = objs
        for o in objs:
            counts[o["type"]] = counts.get(o["type"], 0) + 1
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "n": args.n, "image_ids": eval_ids}, indent=2))
    gt_out = ROOT / args.gt_output
    gt_out.write_text(json.dumps(gt, indent=2))
    print(f"eval split: {len(eval_ids)} images -> {out}")
    print(f"ground truth: {sum(len(v) for v in gt.values())} objects -> {gt_out}")
    print("objects by class:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
