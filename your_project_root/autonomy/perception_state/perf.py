# autonomy/perception_state/perf.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List
import time
import random

@dataclass
class PerfSnapshot:
    t: float
    yolo_fps: float
    yolo_latency_ms: float
    camera_fps: float
    dropped_frames: int
    queue_latency_ms: float
    cpu_pct: float
    ram_pct: float
    temp_c: float
    bottleneck: str

class PerformanceMonitor:
    def __init__(self, drone_id: int):
        self.drone_id = drone_id

        self.yolo_times = deque(maxlen=120)
        self.camera_times = deque(maxlen=120)
        self.yolo_latencies_ms = deque(maxlen=240)
        self.queue_latencies_ms = deque(maxlen=240)

        self.dropped_frames = 0

        # Simulated onboard resource state.
        self.cpu_pct = 20.0
        self.ram_pct = 30.0
        self.temp_c = 42.0

    def record_yolo(self, t: float, latency_ms: float):
        self.yolo_times.append(t)
        self.yolo_latencies_ms.append(latency_ms)

    def record_camera_frame(self, t: float):
        self.camera_times.append(t)

    def record_dropped_frame(self):
        self.dropped_frames += 1

    def record_queue_latency(self, latency_ms: float):
        self.queue_latencies_ms.append(latency_ms)

    def estimate_rate(self, times: deque, now: float, window_s: float = 2.0) -> float:
        if not times:
            return 0.0
        recent = [x for x in times if now - x <= window_s]
        if len(recent) < 2:
            return 0.0
        return len(recent) / float(window_s)

    def update_resources(self, yolo_hz: float):
        """
        Simulated Raspberry Pi resource model.
        Can be replaced with psutil/vcgencm on real hardware.
        """
        target_cpu = 18.0 + yolo_hz * 4.5 + random.uniform(-3.0, 3.0)
        self.cpu_pct = max(2.0, min(100.0, 0.85 * self.cpu_pct + 0.15 * target_cpu))

        target_ram = 32.0 + yolo_hz * 0.7
        self.ram_pct = max(5.0, min(100.0, 0.9 * self.ram_pct + 0.1 * target_ram))

        target_temp = 38.0 + self.cpu_pct * 0.22
        self.temp_c = 0.95 * self.temp_c + 0.05 * target_temp

    def identify_bottleneck(self) -> str:
        yolo_latency = self.avg_yolo_latency_ms()
        queue_latency = self.avg_queue_latency_ms()

        if self.cpu_pct > 90.0:
            return "CPU_SATURATION"
        if yolo_latency > 180.0:
            return "YOLO_LATENCY"
        if queue_latency > 80.0:
            return "QUEUEING"
        if self.temp_c > 78.0:
            return "THERMAL"
        return "NONE"

    def avg_yolo_latency_ms(self) -> float:
        if not self.yolo_latencies_ms:
            return 0.0
        return sum(self.yolo_latencies_ms) / len(self.yolo_latencies_ms)

    def avg_queue_latency_ms(self) -> float:
        if not self.queue_latencies_ms:
            return 0.0
        return sum(self.queue_latencies_ms) / len(self.queue_latencies_ms)

    def snapshot(self, now: float, yolo_hz: float) -> PerfSnapshot:
        self.update_resources(yolo_hz)

        return PerfSnapshot(
            t=now,
            yolo_fps=self.estimate_rate(self.yolo_times, now),
            yolo_latency_ms=self.avg_yolo_latency_ms(),
            camera_fps=self.estimate_rate(self.camera_times, now),
            dropped_frames=self.dropped_frames,
            queue_latency_ms=self.avg_queue_latency_ms(),
            cpu_pct=self.cpu_pct,
            ram_pct=self.ram_pct,
            temp_c=self.temp_c,
            bottleneck=self.identify_bottleneck()
        )