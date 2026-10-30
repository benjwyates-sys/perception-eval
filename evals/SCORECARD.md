# Scorecard — perception-eval, sessions 1–7

Setup: 500-image eval split (seed 42) carved from KITTI's 7,481 labeled
training images; YOLOv8n off the shelf, no training anywhere; IoU 0.5;
evaluated groups Car, Pedestrian, Cyclist in v1, plus TruckVan and Tram
in v2. Per-image results in
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

## Session 4: mapping v2 + per-group slices

Mapping v2 (see README): cyclist unions, TruckVan and Tram groups,
Misc unscored. Slices now per group. Run with `evaluate.py` default;
`--mapping v1` reproduces session 2.

| Group | TP | FP | FN | Precision | Recall | AP | n_gt |
|---|---|---|---|---|---|---|---|
| Car | 1375 | 754 | 472 | 0.646 | 0.744 | 0.680 | 1847 |
| Pedestrian | 163 | 192 | 163 | 0.459 | 0.500 | 0.405 | 326 |
| Cyclist | 17 | 52 | 97 | 0.246 | 0.149 | 0.095 | 114 |
| TruckVan | 56 | 24 | 176 | 0.700 | 0.241 | 0.194 | 232 |
| Tram | 9 | 35 | 23 | 0.205 | 0.281 | 0.120 | 32 |

mAP: 0.299 — NOT comparable to session 2's 0.356. The mean is now over
five groups instead of three, including two hard new ones; Car is
identical, Pedestrian AP rose, Cyclist AP rose 31×. The drop is
composition, not decay.

### What v2 proved, and what it exposed

- Cyclist AP 0.003 → 0.095, TP 2 → 17: the union mechanism works and
  session 3's diagnosis is confirmed. But 97 of 114 cyclists are still
  missed, and fully visible cyclists are missed 83% of the time
  (occlusion=0, n=75) — so the remaining failure is detection, not
  matching. Either the person/bicycle components aren't detected, or
  unions still fall under 0.5 IoU.
- Pedestrian AP 0.386 → 0.405: 19 rider false positives vanished, but
  7 true positives vanished too — greedy IoU > 0 pairing consumes
  genuine pedestrians standing near bikes. Fix: match Pedestrian
  ground truth first, then form unions from leftovers.
- TruckVan precision 0.700 but recall 0.241: most vans are detected as
  cars, and car predictions can't match TruckVan ground truth. The
  recall number measures vocabulary coverage, not detector quality.
  Tram (n=32) is too small a sample for firm conclusions.
- The 25–40px height band is a cliff across classes: Cyclist miss rate
  1.000 (n=27), Pedestrian 0.955 (n=44), TruckVan 0.947 (n=75). Small
  objects, not occluded ones, are this nano detector's real weakness.
- Ignored ground truth is now just Misc (64); truck/bus/train left the
  ignored-prediction list, as they should under v2.

## Session 5: IoU threshold sweep (calibration + optimization)

Sweep of the IoU matching bar, mapping v2, 0.3–0.7 (`sweep_iou.py`;
results in `evals/iou_sweep.json`). The IoU threshold is an evaluation
knob (scoring strictness), not a deployment knob — the deployment dial
is the confidence threshold (session 6).

| IoU | Car | Pedestrian | Cyclist | TruckVan | Tram | mAP |
|---|---|---|---|---|---|---|
| 0.3 | 0.750 | 0.469 | 0.103 | 0.198 | 0.129 | 0.330 |
| 0.4 | 0.724 | 0.452 | 0.103 | 0.198 | 0.129 | 0.321 |
| 0.5 | 0.680 | 0.405 | 0.095 | 0.194 | 0.120 | 0.299 |
| 0.6 | 0.596 | 0.344 | 0.063 | 0.176 | 0.102 | 0.256 |
| 0.7 | 0.483 | 0.216 | 0.030 | 0.154 | 0.036 | 0.184 |

### What the sweep showed

- Group ranking Car > Pedestrian > TruckVan > Tram > Cyclist is
  identical at every threshold: the session 3/4 conclusions do not
  depend on the 0.5 bar.
- Cyclist, TruckVan, and Tram precision/recall are identical at 0.3 and
  0.4: those groups have essentially no matches with IoU in [0.3, 0.4).
  Matches either localize decently (0.4+) or miss entirely — unions
  that match are not borderline; cyclist failures are missing
  components, not sloppy unions. Collapse comes only at strict bars.
- Relative AP drop 0.3→0.7: Tram −72%, Cyclist −71%, Pedestrian −54%,
  Car −36%, TruckVan −22%. IoU is a ratio, so pixel error hurts small
  boxes most (pedestrians) and big truck boxes least; Tram's drop is
  noisy (n=32). Car stays best at every threshold, even 0.483 at 0.7.
- F1-optimal threshold is 0.3 for every group — the edge of the swept
  range, not an interior optimum. Every metric here prefers lenient
  scoring, because lowering the bar manufactures true positives.
  Reporting at 0.3 would game the evaluation, not improve the detector.
  Keep the KITTI-standard 0.5 anchor (also near the curve's knee: gentle
  0.3→0.5, steep after) and report the curve alongside it.
- Calibration shape survives at every threshold (accuracy rises with
  confidence throughout), but the whole curve shifts down as the bar
  rises: the 0.5–0.6 confidence bin goes 0.762 (IoU 0.3) → 0.686 (0.5) →
  0.439 (0.7). Session 3's 'well calibrated' verdict is bar-dependent.
  The 0.9–1.0 bin is invariant at 0.964 accuracy at every threshold:
  the most confident predictions are also the best localized.

## Session 6: confidence threshold sweep (operating points)

Sweep of the confidence threshold, IoU 0.5, mapping v2, 0.25–0.8
(`sweep_conf.py`; results in `evals/conf_sweep.json`). Unlike session
5's IoU sweep (a scoring knob), this is a deployment knob: it decides
which boxes survive at all. F1 leads this analysis; AP at each
threshold is a truncated-list AP (the ranked list just ends earlier),
so it falls monotonically and is secondary.

| Conf | Car | Pedestrian | Cyclist | TruckVan | Tram | mAP | n_pred |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.692 | 0.479 | 0.186 | 0.359 | 0.237 | 0.299 | 2931 |
| 0.3 | 0.709 | 0.493 | 0.206 | 0.337 | 0.277 | 0.290 | 2499 |
| 0.4 | 0.714 | 0.520 | 0.189 | 0.261 | 0.262 | 0.263 | 1941 |
| 0.5 | 0.688 | 0.524 | 0.191 | 0.189 | 0.264 | 0.241 | 1521 |
| 0.6 | 0.607 | 0.470 | 0.157 | 0.142 | 0.250 | 0.195 | 1137 |
| 0.7 | 0.485 | 0.377 | 0.115 | 0.082 | 0.195 | 0.139 | 771 |
| 0.8 | 0.321 | 0.182 | 0.050 | 0.050 | 0.154 | 0.074 | 429 |

F1-optimal per group: Car 0.4 (0.714), Pedestrian 0.5 (0.524),
Cyclist 0.3 (0.206), TruckVan 0.25 (0.359), Tram 0.3 (0.277).

### What the sweep showed

- Real interior optima this time, unlike session 5's edge answer:
  Car peaks at 0.4, Pedestrian at 0.5. Raising the bar genuinely trades
  precision for recall, so F1 peaks in the middle.
- Car is a plateau, not a peak: F1 0.709 (0.3), 0.714 (0.4), 0.688
  (0.5). At 0.3 it is almost perfectly balanced (precision 0.708,
  recall 0.709). Any of 0.3–0.5 is defensible for cars.
- Pedestrian's F1 optimum (0.5) and safety optimum differ: F1 is
  0.524 at 0.5 versus 0.520 at 0.4, while recall is 0.417 versus 0.457.
  Moving to 0.4 buys four points of recall for essentially free — and
  F1 weights a false alarm and a miss equally, which safety does not.
  The safety-weighted pedestrian choice is 0.4, arguably 0.3.
- Cyclist F1 is nearly flat across 0.25–0.5 (0.186–0.206): no
  confidence threshold fixes cyclists. From 0.25 to 0.3, cyclist recall
  does not move (0.149 both) while precision jumps 0.246 → 0.333 — the
  low-confidence cyclist predictions being pruned are pure false
  positives. By 0.5 only 22 cyclist predictions remain of 69.
- TruckVan's F1 falls monotonically from 0.25 onward (edge optimum):
  precision is already 0.700 at the default while recall collapses
  (0.241 → 0.026 by 0.8). Thresholding cannot fix a vocabulary problem
  — its misses are vans detected as cars (session 4).
- Tram is flat and noisy (F1 0.237–0.277 across 0.25–0.6, peak 0.3);
  with n=32, build no recommendation on it.
- Global recommendation: 0.3. Mean F1 across groups peaks at 0.3
  (0.404), ahead of 0.25 (0.391), 0.4 (0.389), and 0.5 (0.371). The
  sweep mostly validates YOLO's 0.25 default rather than overturning
  it. Per-group thresholds (0.4/0.5/0.3/0.25/0.3) buy a little but are
  custom post-processing, not YOLO-native — a complexity cost.
- Session 3's 'threshold around 0.5' hint is refined, not refuted:
  0.5 is Pedestrian's F1 optimum and inside Car's plateau, but
  globally 0.3 wins.

## Session 7: pedestrian-first matching (mapping v3) + cyclist diagnostic

Mapping v3 (same grouping as v2; Pedestrian ground truth claims its
person boxes before unions form) plus `diagnose_cyclist.py`, which sorts
each cyclist ground truth by whether person and/or bicycle boxes even
overlap it. v3 run: `evaluate.py --mapping v3 --output
 evals/metrics_v3.json` (v2's `metrics.json` preserved).

| Group | TP | FP | FN | Precision | Recall | AP | n_gt |
|---|---|---|---|---|---|---|---|
| Car | 1375 | 754 | 472 | 0.646 | 0.744 | 0.680 | 1847 |
| Pedestrian | 170 | 192 | 156 | 0.470 | 0.521 | 0.430 | 326 |
| Cyclist | 16 | 53 | 98 | 0.232 | 0.140 | 0.127 | 114 |
| TruckVan | 56 | 24 | 176 | 0.700 | 0.241 | 0.194 | 232 |
| Tram | 9 | 35 | 23 | 0.205 | 0.281 | 0.120 | 32 |

mAP: 0.310 (comparable to v2's 0.299 — same five groups).

### Pedestrian-first: criteria passed

- Pedestrian TP 163 → 170, FN 163 → 156, FP unchanged: all seven
  session-4 TPs recovered, back to v1 levels, while keeping unions.
  Pedestrian AP 0.405 → 0.430.
- Cyclist TP 17 → 16, FP 52 → 53: the mirror hazard cost exactly one
  TP, as predicted (a rider box claimed by a Pedestrian GT at IoU ≥
  0.5, leaving its bike unpaired). Pre-stated criterion was cyclist
  down by less than pedestrian up: passed clearly.
- Cyclist AP still rose, 0.095 → 0.127: the lost TP was low-ranked,
  and removing a low-precision term raises the average of the rest.
- Recommendation: v3 becomes the new default mapping. It dominates v2
  everywhere except that single cyclist TP. (The code default is still
  v2; flipping it is a one-line change.)

### Cyclist diagnostic: detector blindness, measured

Buckets per cyclist ground truth (present = a box of that class
 overlaps the GT box at IoU > 0; matched = under v2, IoU 0.5):

| Bucket | n | matched | missed |
|---|---|---|---|
| both | 20 | 17 | 3 |
| person_only | 33 | 0 | 33 |
| bike_only | 1 | 0 | 1 |
| neither | 60 | 0 | 60 |

- Of the 97 v2 cyclist misses, 94 lack at least one component: 60
  neither (62%), 33 person-only (34%), 1 bike-only. The residual
  matching problem is 3 cases (both present but missed: 1× occlusion
  0, 2× occlusion 1 — partly-occluded unions failing).
- Below 40px the detector is completely blind: all 27 cyclists in the
  25–40 band and all 3 under 25 have neither component. No matching or
  threshold work touches this; it is a resolution/small-object story.
- person_only is all 40+ and mostly fully visible (24 of 33 at
  occlusion 0): the rider is found, the bike is not, on large clear
  cyclists. Hypothesis: COCO's bicycle class is trained mostly on
  standalone/parked bikes, and a bike with a rider on it (frame
  occluded by legs and torso) looks different enough to be skipped.
- Caveats: buckets use IoU > 0, so adjacent-but-non-overlapping boxes
  count as absent; the diagnostic sees saved predictions only (conf
  ≥ 0.25). A low-confidence predict rerun is the cheap check on
  whether 'neither' shrinks.

## Caveats

- 500 images, one nano detector, IoU 0.5 anchor with a 0.3–0.7
  sensitivity sweep: this is a baseline characterization, not a
  leaderboard claim.
- Session 2's slices were pooled across groups; class mix confounded
  the truncation slice (see finding 4). Session 4 slices are per group.
- Mapping v3 covers all classes except Misc, but has known limits:
  pedestrian-first still allows a rider box to be claimed by a
  Pedestrian GT at IoU ≥ 0.5 (cost 1 cyclist TP in session 7), and
  TruckVan/Tram recall measures vocabulary coverage (vans detected
  as cars) more than detector quality.
- The gallery (20 images) is qualitative; the numbers above are the claim.

## Next

- Adopt v3 as the default mapping (one-line change in evaluate.py).
- Low-confidence predict rerun (below 0.25) to check whether the
  diagnostic's 'neither' bucket shrinks.

## Commands

```
uv run python scripts/evaluate.py   # metrics.json + printed table
uv run python scripts/evaluate.py --mapping v1  # session-2 mapping, for comparison
uv run python scripts/sweep_iou.py  # IoU 0.3–0.7 sweep -> evals/iou_sweep.json
uv run python scripts/sweep_conf.py  # conf 0.25–0.8 sweep -> evals/conf_sweep.json
uv run python scripts/evaluate.py --mapping v3 --output evals/metrics_v3.json  # v3 table
uv run python scripts/diagnose_cyclist.py  # component buckets -> evals/cyclist_diagnostic.json
uv run python scripts/visualize.py  # viz/index.html gallery
```
