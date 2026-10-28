# perception-eval

Evaluate an off-the-shelf object detector on KITTI the way a perception
V&V engineer would: metrics, failure slices, and confidence calibration —
not just one aggregate number. No model training anywhere in this repo;
the detector is a fixed baseline and the harness is the project.

## Session 1: data + baseline predictions

```
cd ~/Downloads/perception-eval
uv sync
uv run python scripts/fetch_kitti.py      # 12GB image download, one-time
uv run python scripts/make_eval_split.py  # 500-image held-out eval split
uv run python scripts/predict.py          # YOLOv8n over the eval images
```

Outputs:

- `data/training/image_2/` — 7,481 KITTI left-camera images (gitignored)
- `data/training/label_2/` — ground-truth labels (gitignored)
- `evals/eval_split.json` — eval image ids + seed (committed)
- `evals/ground_truth.json` — parsed ground truth for the eval images (committed)
- `evals/predictions.json` — raw YOLO predictions per eval image (committed)

Notes:

- KITTI's test-set labels are not public, so the eval split is carved out
  of the labeled training images (seeded sample, seed 42, n=500).
- `predict.py` downloads yolov8n weights (~6MB) on first run and runs on
  CPU; 500 images takes a while, let it run in the background.
- KITTI is a research benchmark; see the KITTI site for its terms.

## What's next

- Session 2: metrics harness — IoU matching, mAP per class, slices by
  distance/size/occlusion/truncation, calibration curves.
- Session 3: failure analysis + scorecard.
