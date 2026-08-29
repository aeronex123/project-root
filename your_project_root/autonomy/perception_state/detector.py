# autonomy/perception_state/detector.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import random
import time

from .camera import CameraFrame
from .config import YoloConfig, CameraConfig, AdaptiveInferenceConfig
from .common_math import clamp

@dataclass
class Detection:
    t: float
    drone_id: int
    label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    image_w: int
    image_h: int
    detector_name: str
    ai_mode: str
    meta: dict = field(default_factory=dict)

    @property
    def center_px(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)

    @property
    def bbox_w_px(self) -> float:
        return max(1.0, abs(self.bbox_xyxy[2] - self.bbox_xyxy[0]))

    @property
    def bbox_h_px(self) -> float:
        return max(1.0, abs(self.bbox_xyxy[3] - self.bbox_xyxy[1]))


class BaseDetector:
    name = "BaseDetector"
    ai_mode = "UNSPECIFIED"

    def detect(self, frame: CameraFrame) -> List[Detection]:
        raise NotImplementedError


class UltralyticsYOLODetector(BaseDetector):
    """
    Real local Ultralytics YOLO interface.

    This only runs if:
      - a local model file is supplied,
      - ultralytics is installed,
      - inference runs on onboard CPU / local accelerator.

    No cloud inference.
    No laptop inference.
    No external AI server.
    """

    name = "UltralyticsYOLO"
    ai_mode = "REAL YOLO"

    def __init__(self, cfg: YoloConfig, drone_id: int):
        self.cfg = cfg
        self.drone_id = drone_id

        try:
            from ultralytics import YOLO
            self.model = YOLO(cfg.model_path)
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def detect(self, frame: CameraFrame) -> List[Detection]:
        if not self.available or frame.image is None:
            return []

        t0 = time.perf_counter()

        results = self.model(
            frame.image,
            imgsz=max(frame.image.shape[:2]) if frame.image.ndim == 3 else 640,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            device=self.cfg.device,
            verbose=False
        )

        out: List[Detection] = []

        for res in results:
            if res.boxes is None:
                continue

            for box in res.boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x0, y0, x1, y1 = map(float, box.xyxy[0])

                    label = self._map_label(res.names.get(cls_id, "OPTIONAL_UNKNOWN_OBJECT"))

                    out.append(Detection(
                        t=frame.t,
                        drone_id=frame.drone_id,
                        label=label,
                        confidence=conf,
                        bbox_xyxy=(x0, y0, x1, y1),
                        image_w=int(frame.image.shape[1]),
                        image_h=int(frame.image.shape[0]),
                        detector_name=self.name,
                        ai_mode=self.ai_mode,
                        meta={
                            "latency_ms": (time.perf_counter() - t0) * 1000.0,
                            "raw_class_id": cls_id
                        }
                    ))
                except Exception:
                    continue

        return out

    def _map_label(self, raw: str) -> str:
        raw = raw.upper()
        if "MINE" in raw and "CUE" in raw:
            return "MINE_SURFACE_CUE"
        if "MINE" in raw:
            return "MINE_SURFACE"
        if "HUMAN" in raw or "PERSON" in raw:
            return "HUMAN"
        if "OBSTACLE" in raw:
            return "OBSTACLE"
        return "OPTIONAL_UNKNOWN_OBJECT"


class SimulationFallbackDetector(BaseDetector):
    """
    Development fallback detector.

    IMPORTANT:
      This is NOT claimed to be YOLO.
      It is a synthetic perception source for simulator bring-up.
    """

    name = "DevFallbackDetector"
    ai_mode = "SIMULATION FALLBACK"

    def __init__(self, cfg: YoloConfig, failure_cfg, drone_id: int):
        self.cfg = cfg
        self.failure_cfg = failure_cfg
        self.drone_id = drone_id

    def detect(self, frame: CameraFrame) -> List[Detection]:
        out: List[Detection] = []

        # False positive generator
        if random.random() < self.failure_cfg.yolo_false_positive_rate:
            w = float(frame.image.shape[1]) if frame.image is not None else 640.0
            h = float(frame.image.shape[0]) if frame.image is not None else 640.0
            x0 = random.uniform(0.1 * w, 0.8 * w)
            y0 = random.uniform(0.1 * h, 0.8 * h)
            x1 = x0 + random.uniform(20, 80)
            y1 = y0 + random.uniform(20, 80)

            out.append(Detection(
                t=frame.t,
                drone_id=self.drone_id,
                label=random.choice(["MINE_SURFACE", "MINE_SURFACE_CUE", "OBSTACLE", "OPTIONAL_UNKNOWN_OBJECT"]),
                confidence=random.uniform(0.28, 0.72),
                bbox_xyxy=(x0, y0, x1, y1),
                image_w=int(w),
                image_h=int(h),
                detector_name=self.name,
                ai_mode=self.ai_mode,
                meta={"false_positive_injected": True}
            ))

        # Synthetic object detections
        for obj in frame.synthetic_objects:
            # Expected obj fields:
            # {
            #   "label": "MINE_SURFACE" | "MINE_SURFACE_CUE" | "HUMAN" | "OBSTACLE" | ...,
            #   "bbox_xyxy": (x0,y0,x1,y1),
            #   "visibility": 0..1,
            #   "occluded": bool
            # }

            if obj.get("occluded", False):
                if random.random() < 0.8:
                    continue

            # False negative model
            if random.random() < self.failure_cfg.yolo_false_negative_rate:
                continue

            visibility = float(obj.get("visibility", 0.8))
            blur_penalty = clamp(float(frame.blur_score), 0.0, 1.0) * 0.35
            texture_bonus = clamp(float(frame.texture_score), 0.0, 1.0) * 0.10

            detect_prob = clamp(visibility - blur_penalty + texture_bonus, 0.05, 0.98)
            if random.random() > detect_prob:
                continue

            conf = clamp(random.gauss(0.55 + 0.35 * visibility, 0.08), 0.15, 0.98)

            x0, y0, x1, y1 = obj["bbox_xyxy"]
            jitter = 2.5
            x0 += random.gauss(0.0, jitter)
            y0 += random.gauss(0.0, jitter)
            x1 += random.gauss(0.0, jitter)
            y1 += random.gauss(0.0, jitter)

            out.append(Detection(
                t=frame.t,
                drone_id=self.drone_id,
                label=str(obj.get("label", "OPTIONAL_UNKNOWN_OBJECT")),
                confidence=float(conf),
                bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                image_w=int(frame.image.shape[1]) if frame.image is not None else 640,
                image_h=int(frame.image.shape[0]) if frame.image is not None else 640,
                detector_name=self.name,
                ai_mode=self.ai_mode,
                meta={
                    "synthetic_source": True,
                    "visibility": visibility
                }
            ))

        return out


def create_detector(yolo_cfg: YoloConfig, failure_cfg, drone_id: int) -> BaseDetector:
    """
    Chooses local real YOLO if available; otherwise development fallback.
    """
    import os

    use_real = False
    if yolo_cfg.model_path and os.path.exists(yolo_cfg.model_path):
        real = UltralyticsYOLODetector(yolo_cfg, drone_id)
        if real.available:
            use_real = True
            print(f"[DRONE {drone_id}] AI MODE: REAL YOLO")
            return real

    if not use_real:
        print(f"[DRONE {drone_id}] AI MODE: SIMULATION FALLBACK")
        return SimulationFallbackDetector(yolo_cfg, failure_cfg, drone_id)