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

## Session 2: metrics harness + visualization

```
uv run python scripts/evaluate.py   # IoU matching, mAP, slices, calibration
uv run python scripts/visualize.py  # annotated images + viz/index.html gallery
```

Outputs:

- `evals/metrics.json` — per-class precision/recall/AP, miss-rate slices by
  occlusion/truncation/height, calibration bins, ignored-class counts.
- `viz/` — annotated images (green = ground truth, blue = true positive,
  red = false positive, gray = unevaluated ground truth, yellow = ignored
  prediction) + gallery HTML (gitignored, regenerable).

Evaluation classes are Car, Pedestrian, and Cyclist (KITTI's main
benchmark trio). COCO→KITTI mapping: car→Car, person→Pedestrian (GT
Pedestrian and Person_sitting grouped — the detector can't tell them
apart), bicycle→Cyclist. Other COCO predictions and Van/Truck/Tram/Misc
ground truth are counted as ignored, not scored; the tram gap (no COCO
equivalent) is a known expected miss.

## Session 3: failure analysis + scorecard

Findings in `evals/SCORECARD.md`: mAP 0.356 — Car AP 0.680, Pedestrian
0.386, Cyclist 0.003. The cyclist collapse is a mapping artifact
(rider+bike union box vs bike-only box); the real degradation is clean
occlusion and size gradients; calibration is good and the confidence
score is a usable operating dial.

## What's next

- Mapping v2: rider+bike cyclist unions, Van/Truck/Tram scoring.
- Per-group slices (current slices are pooled across classes).
- Threshold sweep for the operating point.
