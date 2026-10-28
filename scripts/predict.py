"""Run an off-the-shelf YOLO detector over the eval split (no training)."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evals/eval_split.json")
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--output", default="evals/predictions.json")
    args = ap.parse_args()
    from ultralytics import YOLO

    split = json.loads((ROOT / args.split).read_text())
    image_dir = ROOT / "data" / "training" / "image_2"
    model = YOLO(args.model)  # downloads weights on first run
    out = {}
    for i, iid in enumerate(split["image_ids"]):
        results = model(str(image_dir / f"{iid}.png"), verbose=False)
        preds = []
        for box in results[0].boxes:
            preds.append({
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
                "score": float(box.conf[0]),
                "class": model.names[int(box.cls[0])],
            })
        out[iid] = preds
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(split['image_ids'])} images done")
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    total = sum(len(v) for v in out.values())
    print(f"predictions: {total} boxes over {len(out)} images -> {out_path}")


if __name__ == "__main__":
    main()
