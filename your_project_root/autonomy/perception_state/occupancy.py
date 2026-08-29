# autonomy/perception_state/occupancy.py
from __future__ import annotations
from enum import IntEnum
from typing import List
import numpy as np
import math

from .config import OccupancyConfig

class CellType(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    MINE = 3
    EXCLUSION = 4
    SAFE_PATH = 5
    HUMAN = 6

class OccupancyGrid:
    def __init__(self, cfg: OccupancyConfig, origin_xy=(0.0, 0.0)):
        self.cfg = cfg
        self.origin_xy = np.array(origin_xy, dtype=float)

        self.nx = int(cfg.width_m / cfg.resolution_m)
        self.ny = int(cfg.height_m / cfg.resolution_m)

        self.log_odds = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.semantic = np.full((self.nx, self.ny), int(CellType.UNKNOWN), dtype=np.uint8)

    def memory_bytes(self) -> int:
        return int(self.log_odds.nbytes + self.semantic.nbytes)

    def world_to_index(self, xy: np.ndarray):
        local = (np.asarray(xy, dtype=float)[:2] - self.origin_xy) / self.cfg.resolution_m
        ix = int(math.floor(local[0] + self.nx * 0.5))
        iy = int(math.floor(local[1] + self.ny * 0.5))
        return ix, iy

    def index_to_world(self, ix: int, iy: int) -> np.ndarray:
        x = (ix - self.nx * 0.5) * self.cfg.resolution_m + self.origin_xy[0]
        y = (iy - self.ny * 0.5) * self.cfg.resolution_m + self.origin_xy[1]
        return np.array([x, y], dtype=float)

    def inside(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def update_cell(self, ix: int, iy: int, increment: float, semantic: CellType):
        if not self.inside(ix, iy):
            return

        self.log_odds[ix, iy] = np.clip(
            self.log_odds[ix, iy] + increment,
            self.cfg.clamp_min,
            self.cfg.clamp_max
        )

        # Semantic priority:
        # MINE / EXCLUSION dominate obstacles/free.
        current = CellType(self.semantic[ix, iy])

        if semantic == CellType.MINE:
            self.semantic[ix, iy] = int(CellType.MINE)
        elif semantic == CellType.EXCLUSION:
            if current not in (CellType.MINE,):
                self.semantic[ix, iy] = int(CellType.EXCLUSION)
        elif semantic == CellType.HUMAN:
            if current not in (CellType.MINE, CellType.EXCLUSION):
                self.semantic[ix, iy] = int(CellType.HUMAN)
        elif semantic == CellType.OBSTACLE:
            if current not in (CellType.MINE, CellType.EXCLUSION, CellType.HUMAN):
                self.semantic[ix, iy] = int(CellType.OBSTACLE)
        elif semantic == CellType.FREE:
            if current == CellType.UNKNOWN:
                self.semantic[ix, iy] = int(CellType.FREE)
        elif semantic == CellType.SAFE_PATH:
            if current in (CellType.FREE, CellType.UNKNOWN):
                self.semantic[ix, iy] = int(CellType.SAFE_PATH)

    def mark_circle(
        self,
        center_xy: np.ndarray,
        radius_m: float,
        increment: float,
        semantic: CellType
    ):
        cx, cy = center_xy
        r_cells = int(math.ceil(radius_m / self.cfg.resolution_m))

        ix0, iy0 = self.world_to_index(center_xy)

        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                ix = ix0 + dx
                iy = iy0 + dy
                if not self.inside(ix, iy):
                    continue

                w = self.index_to_world(ix, iy)
                dist = float(np.linalg.norm(w[:2] - np.asarray([cx, cy])))
                if dist <= radius_m:
                    self.update_cell(ix, iy, increment, semantic)

    def inflate_mine(self, pos2: np.ndarray, mine_radius_m: float, clearance_m: float):
        """
        Mine forbidden region:
          physical mine radius + 1.0 m clearance

        Circular/Euclidean inflation.
        """
        self.mark_circle(
            center_xy=pos2,
            radius_m=mine_radius_m,
            increment=self.cfg.mine_increment,
            semantic=CellType.MINE
        )

        self.mark_circle(
            center_xy=pos2,
            radius_m=mine_radius_m + clearance_m,
            increment=self.cfg.mine_increment,
            semantic=CellType.EXCLUSION
        )

    def mark_obstacle_point(self, pos2: np.ndarray, radius_m: float = 0.10):
        self.mark_circle(
            center_xy=pos2,
            radius_m=radius_m,
            increment=self.cfg.obstacle_increment,
            semantic=CellType.OBSTACLE
        )

    def mark_human_point(self, pos2: np.ndarray, radius_m: float = 0.35):
        self.mark_circle(
            center_xy=pos2,
            radius_m=radius_m,
            increment=self.cfg.human_increment,
            semantic=CellType.HUMAN
        )

    def mark_free_point(self, pos2: np.ndarray, radius_m: float = 0.15):
        self.mark_circle(
            center_xy=pos2,
            radius_m=radius_m,
            increment=self.cfg.free_increment,
            semantic=CellType.FREE
        )