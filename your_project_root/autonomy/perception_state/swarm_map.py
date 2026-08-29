# autonomy/perception_state/swarm_map.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import random
import time

from .config import SwarmCommsConfig

@dataclass
class MapDelta:
    source_drone: int
    t: float
    seq: int
    kind: str       # MINE, OBSTACLE, EXPLORED, PATH
    payload: dict = field(default_factory=dict)

class SwarmChannel:
    """
    Simulated local swarm datalink.

    No cloud.
    No external AI server.
    """

    def __init__(self, cfg: SwarmCommsConfig):
        self.cfg = cfg
        self.in_flight: List[tuple] = []
        self.packet_count = 0
        self.lost_count = 0
        self.latencies_ms: List[float] = []

    def send(self, delta: MapDelta, now: float):
        self.packet_count += 1

        if random.random() < self.cfg.packet_loss_prob:
            self.lost_count += 1
            return

        latency_ms = max(
            1.0,
            random.gauss(self.cfg.base_latency_ms, self.cfg.jitter_ms)
        )
        self.latencies_ms.append(latency_ms)

        deliver_at = now + latency_ms * 1e-3
        self.in_flight.append((deliver_at, delta))

    def receive(self, now: float) -> List[MapDelta]:
        ready = []
        remaining = []

        for deliver_at, delta in self.in_flight:
            if now >= deliver_at:
                ready.append(delta)
            else:
                remaining.append((deliver_at, delta))

        self.in_flight = remaining
        return ready

    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def packet_loss_rate(self) -> float:
        if self.packet_count == 0:
            return 0.0
        return self.lost_count / float(self.packet_count)


class DroneMapSync:
    """
    Each drone maintains LOCAL_MAP and receives REMOTE_MAP_UPDATES.
    Only compact deltas are transmitted.
    """

    def __init__(self, drone_id: int, cfg: SwarmCommsConfig, channel: SwarmChannel):
        self.drone_id = drone_id
        self.cfg = cfg
        self.channel = channel
        self.tx_seq = 0
        self.last_rx_seq = {}

    def publish_mine(self, t: float, mine_id: str, pos2, cov2, confidence: float, label: str):
        delta = MapDelta(
            source_drone=self.drone_id,
            t=t,
            seq=self.tx_seq,
            kind="MINE",
            payload={
                "mine_id": mine_id,
                "pos2": list(map(float, pos2)),
                "cov2": cov2.tolist(),
                "confidence": float(confidence),
                "label": label
            }
        )
        self.tx_seq += 1
        self.channel.send(delta, t)

    def publish_obstacle(self, t: float, pos2, radius: float):
        delta = MapDelta(
            source_drone=self.drone_id,
            t=t,
            seq=self.tx_seq,
            kind="OBSTACLE",
            payload={
                "pos2": list(map(float, pos2)),
                "radius": float(radius)
            }
        )
        self.tx_seq += 1
        self.channel.send(delta, t)

    def publish_explored(self, t: float, cells: list):
        # Compact run-length or sparse cell list recommended.
        delta = MapDelta(
            source_drone=self.drone_id,
            t=t,
            seq=self.tx_seq,
            kind="EXPLORED",
            payload={"cells": cells[:256]}
        )
        self.tx_seq += 1
        self.channel.send(delta, t)

    def publish_path(self, t: float, path: list):
        delta = MapDelta(
            source_drone=self.drone_id,
            t=t,
            seq=self.tx_seq,
            kind="PATH",
            payload={"path": path}
        )
        self.tx_seq += 1
        self.channel.send(delta, t)

    def receive(self, now: float) -> List[MapDelta]:
        deltas = self.channel.receive(now)

        out = []
        for d in deltas:
            if d.source_drone == self.drone_id:
                continue

            last = self.last_rx_seq.get(d.source_drone, -1)
            if d.seq <= last:
                continue

            self.last_rx_seq[d.source_drone] = d.seq
            out.append(d)

        return out