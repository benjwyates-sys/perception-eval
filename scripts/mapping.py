"""COCO -> KITTI class mapping, versions 1 and 2, + cyclist union boxes.

v1 (session 2): Car, Pedestrian, Cyclist only. person -> Pedestrian (GT
Pedestrian + Person_sitting grouped), car -> Car, bicycle -> Cyclist.
Van/Truck/Tram/Misc ground truth and other COCO predictions ignored.

v2 (session 4): adds cyclist unions and two groups. A bicycle box paired
with its most-overlapping person box (any overlap, IoU > 0) becomes a
union box scored as a Cyclist candidate; the consumed person box no
longer scores as a pedestrian, and the paired bicycle box is consumed
too. Unpaired bicycle boxes stay Cyclist candidates as before. New
groups: TruckVan (GT Truck + Van, preds truck) and Tram (GT Tram, preds
bus + train). Misc stays unscored.

v3 (session 7): grouping identical to v2. The difference is matching
order, which lives in evaluate.py, not here: Pedestrian ground truth
claims its person boxes BEFORE unions form, so a genuine pedestrian
standing near a bike can no longer be consumed by a union. Unions form
only from leftover person boxes."""
from metrics import iou

GT_MAP_V1 = {"Car": "Car", "Pedestrian": "Pedestrian", "Person_sitting": "Pedestrian", "Cyclist": "Cyclist"}
PRED_MAP_V1 = {"car": "Car", "person": "Pedestrian", "bicycle": "Cyclist"}
GROUPS_V1 = ["Car", "Pedestrian", "Cyclist"]

GT_MAP_V2 = dict(GT_MAP_V1, Truck="TruckVan", Van="TruckVan", Tram="Tram")
PRED_MAP_V2 = dict(PRED_MAP_V1, truck="TruckVan", bus="Tram", train="Tram")
GROUPS_V2 = ["Car", "Pedestrian", "Cyclist", "TruckVan", "Tram"]

MAPS = {"v1": (GT_MAP_V1, PRED_MAP_V1, GROUPS_V1), "v2": (GT_MAP_V2, PRED_MAP_V2, GROUPS_V2),
        "v3": (GT_MAP_V2, PRED_MAP_V2, GROUPS_V2)}


def build_gt_groups(gt_objs, version):
    gt_map, _, groups = MAPS[version]
    by_group = {g: [] for g in groups}
    ignored = []
    for o in gt_objs:
        g = gt_map.get(o["type"])
        if g is None:
            ignored.append(o)
        else:
            by_group[g].append(o)
    return by_group, ignored


def _cyclist_unions(person_boxes, bicycle_boxes):
    """Pair each bicycle box with its most-overlapping unused person box.
    Returns (union_preds, used_person_idx, used_bicycle_idx)."""
    used_persons = set()
    used_bicycles = set()
    unions = []
    for bi in sorted(range(len(bicycle_boxes)), key=lambda i: -bicycle_boxes[i]["score"]):
        box = bicycle_boxes[bi]
        best, best_iou = None, 0.0
        for i, p in enumerate(person_boxes):
            if i in used_persons:
                continue
            v = iou(box["bbox"], p["bbox"])
            if v > best_iou:
                best, best_iou = i, v
        if best is not None:
            p = person_boxes[best]
            used_persons.add(best)
            used_bicycles.add(bi)
            unions.append({
                "bbox": [min(box["bbox"][0], p["bbox"][0]), min(box["bbox"][1], p["bbox"][1]),
                         max(box["bbox"][2], p["bbox"][2]), max(box["bbox"][3], p["bbox"][3])],
                "score": max(box["score"], p["score"]),
                "class": "bicycle",
                "union": True,
            })
    return unions, used_persons, used_bicycles


def build_pred_groups(preds, version):
    _, pred_map, groups = MAPS[version]
    by_group = {g: [] for g in groups}
    ignored = []
    if version in ("v2", "v3"):
        persons = [p for p in preds if p["class"] == "person"]
        bicycles = [p for p in preds if p["class"] == "bicycle"]
        unions, used_p, used_b = _cyclist_unions(persons, bicycles)
        by_group["Cyclist"].extend(unions)
        by_group["Cyclist"].extend([b for i, b in enumerate(bicycles) if i not in used_b])
        rest = [p for p in preds if p["class"] not in ("person", "bicycle")]
        persons = [p for i, p in enumerate(persons) if i not in used_p]
        for p in persons + rest:
            g = pred_map.get(p["class"])
            if g is None:
                ignored.append(p)
            else:
                by_group[g].append(p)
    else:
        for p in preds:
            g = pred_map.get(p["class"])
            if g is None:
                ignored.append(p)
            else:
                by_group[g].append(p)
    return by_group, ignored
