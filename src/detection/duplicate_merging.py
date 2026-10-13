"""Removes duplicate detections when two models find the same vehicle."""

import numpy as np

from src.detection.vehicle_types import CLASS_INDEX, CONFIDENCE, PERSON_CLASS_INDEX, X1, X2, Y1, Y2

IOU_THRESHOLD = 0.5  # a box is dropped when its IoU with a more confident box exceeds this
CONTAINMENT_THRESHOLD = 0.8  # ...or when this share of the smaller box lies inside a more confident box
EPSILON = 1e-9  # avoids dividing by zero for degenerate boxes


def merge_duplicates(detections: np.ndarray) -> np.ndarray:
    """Keeps the most confident box of each group of overlapping boxes (class-agnostic among vehicles; people are
    only compared with other people, so a person standing next to a car is not removed).

    Containment (overlap over the smaller box) catches the case where the OBB model's axis-aligned box around a
    diagonal car is much larger than the COCO model's box for the same car, which gives a low IoU.

    Returns:
        The kept detections, sorted by descending confidence.
    """
    if len(detections) < 2:
        return detections

    detections = detections[np.argsort(-detections[:, CONFIDENCE])]
    left, top, right, bottom = detections[:, X1], detections[:, Y1], detections[:, X2], detections[:, Y2]
    areas = (right - left) * (bottom - top)
    is_person = detections[:, CLASS_INDEX] == PERSON_CLASS_INDEX

    is_dropped = np.zeros(len(detections), dtype=bool)
    kept_rows = []
    for current in range(len(detections)):
        if is_dropped[current]:
            continue
        kept_rows.append(current)

        # Compare this box against every less confident box that comes after it.
        later = slice(current + 1, None)
        overlap_width = np.clip(np.minimum(right[current], right[later]) - np.maximum(left[current], left[later]), 0, None)
        overlap_height = np.clip(np.minimum(bottom[current], bottom[later]) - np.maximum(top[current], top[later]), 0, None)
        overlap_area = overlap_width * overlap_height

        iou = overlap_area / (areas[current] + areas[later] - overlap_area + EPSILON)
        containment = overlap_area / (np.minimum(areas[current], areas[later]) + EPSILON)
        is_comparable = is_person[later] == is_person[current]
        is_dropped[later] |= is_comparable & ((iou > IOU_THRESHOLD) | (containment > CONTAINMENT_THRESHOLD))

    return detections[kept_rows]
