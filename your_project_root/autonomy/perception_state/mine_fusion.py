# autonomy/perception_state/mine_fusion.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

from .geoloc import MineObservation
from .config import MineFusionConfig

@dataclass
class LocalMineTrack:
    track_id: str
    label: str
    pos2: np.ndarray
    cov2: np.ndarray
    confidence_sum: float
    obs_count: int
    first_t: float
    last_t: float
    sources: set = field(default_factory=set)
    confirmed: bool = False

    @property
    def avg_conf(self) -> float:
        if self.obs_count == 0:
            return 0.0
        return self.confidence_sum / float(self.obs_count)

    def consistency(self) -> float:
        trace = float(np.trace(self.cov2))
        return math.exp(-trace)

class LocalMineConfirmationEngine:
    """
    Local temporal confirmation.

    A detection becomes a confirmed mine event only after:
      - enough observations,
      - sufficient confidence,
      - spatial consistency.
    """

    def __init__(self, cfg: MineFusionConfig, drone_id: int):
        self.cfg = cfg
        self.drone_id = drone_id
        self.tracks: Dict[str, LocalMineTrack] = {}
        self.next_local_id = 1

    def add_observation(self, obs: MineObservation) -> Optional[str]:
        best_id, best_score = None, 0.0

        for tid, trk in self.tracks.items():
            d = float(np.linalg.norm(trk.pos2 - obs.pos2))
            if d > self.cfg.candidate_gate_m:
                continue

            combined_cov = trk.cov2 + obs.cov2 + np.eye(2) * 1e-6
            diff = trk.pos2 - obs.pos2
            try:
                mahal = float(np.sqrt(diff @ np.linalg.inv(combined_cov) @ diff))
            except np.linalg.LinAlgError:
                mahal = 999.0

            score = math.exp(-mahal)
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is None:
            tid = f"LOCAL_{self.drone_id}_{self.next_local_id:04d}"
            self.next_local_id += 1
            self.tracks[tid] = LocalMineTrack(
                track_id=tid,
                label=obs.label,
                pos2=obs.pos2.copy(),
                cov2=obs.cov2.copy(),
                confidence_sum=obs.confidence,
                obs_count=1,
                first_t=obs.t,
                last_t=obs.t,
                sources={obs.drone_id},
                confirmed=False
            )
            return None

        trk = self.tracks[best_id]

        # Simple covariance-weighted fusion
        P1_inv = np.linalg.pinv(trk.cov2 + np.eye(2)*1e-6)
        P2_inv = np.linalg.pinv(obs.cov2 + np.eye(2)*1e-6)
        P_fused_inv = P1_inv + P2_inv
        P_fused = np.linalg.pinv(P_fused_inv)

        pos_fused = P_fused @ (P1_inv @ trk.pos2 + P2_inv @ obs.pos2)

        trk.pos2 = pos_fused
        trk.cov2 = P_fused
        trk.confidence_sum += obs.confidence
        trk.obs_count += 1
        trk.last_t = obs.t
        trk.sources.add(obs.drone_id)

        if (not trk.confirmed) and self._is_confirmed(trk):
            trk.confirmed = True
            return trk.track_id

        return None

    def _is_confirmed(self, trk: LocalMineTrack) -> bool:
        if trk.obs_count < self.cfg.confirmation_min_obs:
            return False

        if trk.avg_conf < self.cfg.confirmation_conf:
            return False

        if float(np.trace(trk.cov2)) > self.cfg.confirmation_cov_trace_max:
            return False

        return True


@dataclass
class GlobalMine:
    mine_id: str
    label: str
    pos2: np.ndarray
    cov2: np.ndarray
    confidence: float
    obs_count: int
    first_t: float
    last_t: float
    sources: set = field(default_factory=set)
    radius_m: float = 0.15
    clearance_m: float = 1.0

class GlobalMineMap:
    """
    Fused global mine map.

    If Drone 1 and Drone 2 observe the same physical mine,
    this must produce ONE global mine.
    """

    def __init__(self, cfg: MineFusionConfig):
        self.cfg = cfg
        self.mines: Dict[str, GlobalMine] = {}
        self.next_global_id = 1

        # metrics
        self.total_confirmed_events = 0
        self.duplicate_associations = 0
        self.new_mines_created = 0

    def add_confirmed_observation(self, obs: MineObservation) -> Tuple[str, bool]:
        """
        Returns:
          global_mine_id, is_duplicate
        """
        self.total_confirmed_events += 1

        best_id = None
        best_score = -1.0

        for mid, mine in self.mines.items():
            d = float(np.linalg.norm(mine.pos2 - obs.pos2))
            if d > max(self.cfg.duplicate_gate_m, 2.0 * np.sqrt(np.trace(mine.cov2 + obs.cov2))):
                continue

            combined_cov = mine.cov2 + obs.cov2 + np.eye(2) * 1e-6
            diff = mine.pos2 - obs.pos2
            try:
                mahal = float(np.sqrt(diff @ np.linalg.inv(combined_cov) @ diff))
            except np.linalg.LinAlgError:
                mahal = 999.0

            if mahal > self.cfg.duplicate_mahalanobis_gate:
                continue

            score = math.exp(-mahal) * (1.0 + mine.confidence)
            if score > best_score:
                best_score = score
                best_id = mid

        if best_id is None:
            mid = f"MINE_{self.next_global_id:03d}"
            self.next_global_id += 1

            self.mines[mid] = GlobalMine(
                mine_id=mid,
                label=obs.label,
                pos2=obs.pos2.copy(),
                cov2=obs.cov2.copy(),
                confidence=obs.confidence,
                obs_count=1,
                first_t=obs.t,
                last_t=obs.t,
                sources={obs.drone_id},
                radius_m=max(0.08, obs.size_m * 0.5),
                clearance_m=self.cfg.required_clearance_m
            )

            self.new_mines_created += 1
            return mid, False

        mine = self.mines[best_id]

        # Fuse position and covariance
        P1_inv = np.linalg.pinv(mine.cov2 + np.eye(2)*1e-6)
        P2_inv = np.linalg.pinv(obs.cov2 + np.eye(2)*1e-6)
        P_fused_inv = P1_inv + P2_inv
        P_fused = np.linalg.pinv(P_fused_inv)

        pos_fused = P_fused @ (P1_inv @ mine.pos2 + P2_inv @ obs.pos2)

        mine.pos2 = pos_fused
        mine.cov2 = P_fused
        mine.confidence = max(mine.confidence, obs.confidence)
        mine.obs_count += 1
        mine.last_t = obs.t
        mine.sources.add(obs.drone_id)

        self.duplicate_associations += 1
        return mine.mine_id, True

    def duplicate_mine_rate(self) -> float:
        if self.total_confirmed_events == 0:
            return 0.0
        return self.duplicate_associations / float(self.total_confirmed_events)