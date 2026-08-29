# autonomy/perception_state/report.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
import math

@dataclass
class FinalReport:
    detection_precision: float = 0.0
    detection_recall: float = 0.0
    f1_score: float = 0.0

    localization_rmse_m: float = 0.0
    map_accuracy: float = 0.0

    duplicate_mine_rate: float = 0.0
    false_mine_rate: float = 0.0

    minimum_mine_clearance_m: float = 0.0

    yolo_fps_mean: float = 0.0
    yolo_latency_mean_ms: float = 0.0

    cpu_mean_pct: float = 0.0
    ram_mean_pct: float = 0.0

    comm_latency_mean_ms: float = 0.0
    packet_loss_rate: float = 0.0

    mission_time_s: float = 0.0

    failures_visible: List[str] = field(default_factory=list)


def compute_rmse(errors: List[float]) -> float:
    if not errors:
        return 0.0
    arr = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def compute_precision_recall(
    confirmed_mines,
    ground_truth_mines,
    match_radius_m: float = 0.75
):
    """
    confirmed_mines: list of estimated mine positions
    ground_truth_mines: list of true mine positions

    A true positive is a confirmed mine within match_radius_m of a truth mine.
    """
    tp = 0
    matched_truth = set()

    for est in confirmed_mines:
        best_i = None
        best_d = 1e9

        for i, gt in enumerate(ground_truth_mines):
            if i in matched_truth:
                continue
            d = float(np.linalg.norm(np.asarray(est[:2]) - np.asarray(gt[:2])))
            if d < best_d:
                best_d = d
                best_i = i

        if best_i is not None and best_d <= match_radius_m:
            tp += 1
            matched_truth.add(best_i)

    fp = len(confirmed_mines) - tp
    fn = len(ground_truth_mines) - len(matched_truth)

    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1