"""Pure evaluation functions: IoU, greedy matching, AP, calibration bins."""


def iou(box_a, box_b):
    """Intersection over union for [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Greedy matching within one class: predictions in descending score order,
    each takes the highest-IoU unmatched GT box above threshold.
    Returns (tp_flags aligned with pred_boxes, {pred_idx: gt_idx})."""
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_boxes[i]["score"])
    used = set()
    tp = [False] * len(pred_boxes)
    matched = {}
    for i in order:
        best, best_iou = None, iou_threshold
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(pred_boxes[i]["bbox"], g["bbox"])
            if v > best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            tp[i] = True
            matched[i] = best
    return tp, matched


def average_precision(tp_flags_by_score, n_gt):
    """All-point AP: sum over true positives of (recall step * precision)."""
    if n_gt == 0:
        return 0.0
    ap = 0.0
    prev_recall = 0.0
    tp_cum = 0
    for k, is_tp in enumerate(tp_flags_by_score):
        if is_tp:
            tp_cum += 1
            recall = tp_cum / n_gt
            ap += (recall - prev_recall) * (tp_cum / (k + 1))
            prev_recall = recall
    return ap


def calibration_bins(tp_flags, scores, n_bins=10):
    """Per confidence bin: how many predictions, and what fraction were TPs.
    A calibrated detector's bin accuracy tracks the bin's confidence."""
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores) if (lo <= s < hi) or (b == n_bins - 1 and s == 1.0)]
        bins.append({
            "bin": [lo, hi],
            "n": len(idx),
            "accuracy": (sum(tp_flags[i] for i in idx) / len(idx)) if idx else None,
        })
    return bins
