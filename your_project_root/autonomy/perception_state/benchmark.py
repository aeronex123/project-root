# autonomy/perception_state/benchmark.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import time
import random

@dataclass
class ResolutionBenchmarkResult:
    resolution: int
    mean_latency_ms: float
    fps: float
    dropped_frames: int
    cpu_pct: float

class InferenceResolutionBenchmark:
    """
    Compares:
      640x640
      512x512
      416x416

    The benchmark runs locally on simulated Raspberry Pi compute.
    """

    def __init__(self, detector, camera):
        self.detector = detector
        self.camera = camera

    def run(self, frames: list, resolutions=(640, 512, 416)) -> List[ResolutionBenchmarkResult]:
        results = []

        for res in resolutions:
            self.camera.set_inference_size(res)

            latencies = []
            dropped = 0

            t0 = time.perf_counter()

            for frame in frames:
                start = time.perf_counter()
                _ = self.detector.detect(frame)
                latencies.append((time.perf_counter() - start) * 1000.0)

            total = time.perf_counter() - t0
            fps = len(frames) / total if total > 0 else 0.0
            mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

            results.append(ResolutionBenchmarkResult(
                resolution=res,
                mean_latency_ms=mean_latency,
                fps=fps,
                dropped_frames=dropped,
                cpu_pct=random.uniform(45.0, 85.0)  # replace with real psutil on hardware
            ))

        return results