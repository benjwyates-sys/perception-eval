# Scorecard — perception-eval, sessions 1–3

Setup: 500-image eval split (seed 42) carved from KITTI's 7,481 labeled
training images; YOLOv8n off the shelf, no training anywhere; IoU 0.5;
evaluated groups Car, Pedestrian, Cyclist. Per-image results in
`evals/metrics.json`; annotated images in `viz/` (gitignored).

## Results

| Group | TP | FP | FN | Precision | Recall | AP | n_gt |
|---|---|---|---|---|---|---|---|
| Car | 1375 | 754 | 472 | 0.646 | 0.744 | 0.680 | 1847 |
| Pedestrian | 170 | 211 | 156 | 0.446 | 0.521 | 0.386 | 326 |
| Cyclist | 2 | 67 | 112 | 0.029 | 0.018 | 0.003 | 114 |

mAP: 0.356

## Findings

1. **The cyclist number is a mapping artifact, not detector blindness.**
   Cyclist AP is 0.003 with 2 of 114 detected — but the same row shows 67
   bicycle false positives, so the detector fires on bikes plenty. A KITTI
   Cyclist box covers rider *plus* bike as one union box; YOLO's bicycle
   box covers the bike alone, so it falls under the 0.5 IoU bar. The
   rider's `person` box goes to the Pedestrian group, where no Pedestrian
   ground truth exists to match it — one mapping flaw creates a Cyclist
   miss and a Pedestrian false positive at once. Gallery check: a blue
   bicycle box sitting inside a bigger green cyclist box confirms it.
2. **Precision is deflated by unevaluated classes.** Car's 754 false
   positives include car boxes landing on Vans (178) and Trucks (54) —
   ground truth the harness deliberately doesn't score, so a
   correct-looking detection counts as an error. The ignored-prediction
   list shows the detector sees the unscored world fine: truck 80, bus
   22, train 22 against Truck/Van/Tram ground truth, plus 131 traffic
   lights — real objects KITTI simply doesn't annotate. Vocabulary
   mismatch, not blindness.
3. **The real degradation is occlusion and size, and it's clean.**
   Miss rate by occlusion: fully visible 0.198 (n=1175) → partly
   occluded 0.345 (n=641) → largely occluded 0.625 (n=397). By box
   height: under 25px 0.478 (n=291) → 25–40px 0.407 (n=573) → over 40px
   0.259 (n=1423). Both gradients are monotonic and predictable: small
   and occluded objects are where this detector actually fails, and both
   point at fixes (higher resolution, occlusion-aware training) rather
   than a shrug. Occlusion level 3 (unknown, n=74, 0.514) breaks the
   monotonicity slightly; level 3 means occlusion wasn't labeled, not a
   fourth severity step.
4. **Truncation looks backwards — probably composition, not help.**
   Partially truncated objects are missed less often (0.230, n=283)
   than untruncated ones (0.337, n=2004). Truncated objects tend to be
   close, large cars at the image edge, and these slices are pooled
   across classes, so class mix confounds them. Per-group slices would
   separate that out.
5. **Calibration is good, and the confidence score is a usable dial.**
   Overall bins (confidence → fraction correct): 0.2–0.3 → 0.21
   (n=352), 0.3–0.4 → 0.29 (n=457), 0.4–0.5 → 0.44 (n=370), 0.5–0.6 →
   0.68 (n=342), 0.6–0.7 → 0.83 (n=318), 0.7–0.8 → 0.90 (n=326),
   0.8–0.9 → 0.93 (n=333), 0.9–1.0 → 0.96 (n=81). Well calibrated at
   the extremes, mildly underconfident in the middle (the detector is
   more often right than its score claims between 0.5 and 0.9). Bins
   below 0.2 are empty because YOLO's default 0.25 confidence threshold
   means nothing below it ever becomes a prediction. Practical upshot:
   a threshold around 0.5 keeps mostly-right predictions and sheds most
   of the false-positive tail.

## Caveats

- 500 images, one nano detector, one IoU threshold: this is a baseline
  characterization, not a leaderboard claim.
- Slices are pooled across groups; class mix confounds the truncation
  slice (see finding 4) and probably shades the others.
- Mapping v1 is deliberately narrow (3 groups); Van/Truck/Tram/Misc are
  unscored by design, and the cyclist mapping is known-broken (finding 1).
- The gallery (20 images) is qualitative; the numbers above are the claim.

## Next

- Mapping v2: merge rider+bike boxes into cyclist unions; score
  Van/Truck/Tram via truck/bus/train predictions.
- Per-group slices, so occlusion/size effects are separable by class.
- Threshold sweep for the operating point the calibration curve implies.

## Commands

```
uv run python scripts/evaluate.py   # metrics.json + printed table
uv run python scripts/visualize.py  # viz/index.html gallery
```
