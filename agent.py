# Agent dataclass, kinematics, state machine.

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np


class AgentState(enum.Enum):
    IDLE = "idle"
    MOVING = "moving"
    SEARCHING = "searching"
    RETURNING = "returning"


@dataclass
class Agent:
    """A single UAV agent in the swarm."""

    id: int
    pos: np.ndarray            # [x, y] in world coords (pixels)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0       # radians, 0 = east, pi/2 = south
    state: AgentState = AgentState.IDLE

    # tuning
    max_speed: float = 120.0   # px/s
    max_force: float = 200.0   # px/s²
    radius: float = 8.0        # px, used for collision and rendering

    # optional target waypoint
    target: np.ndarray | None = None

    # assigned zone polygon (set by coordinator)
    zone: np.ndarray | None = None

    # ── colours per state ──────────────────────────────────────────
    STATE_COLOURS: dict[AgentState, tuple[int, int, int]] = field(
        default=None, init=False, repr=False,
    )

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.vel = np.asarray(self.vel, dtype=float)
        self.STATE_COLOURS = {
            AgentState.IDLE:      (100, 200, 255),
            AgentState.MOVING:    (80, 255, 80),
            AgentState.SEARCHING: (255, 200, 60),
            AgentState.RETURNING: (255, 100, 100),
        }

    # ── physics ────────────────────────────────────────────────────
    def seek(self, target: np.ndarray) -> np.ndarray:
        """Steering force toward a target point."""
        desired = np.asarray(target, dtype=float) - self.pos
        dist = np.linalg.norm(desired)
        if dist < 1e-6:
            return np.zeros(2)
        desired = desired / dist * self.max_speed
        steer = desired - self.vel
        mag = np.linalg.norm(steer)
        if mag > self.max_force:
            steer = steer / mag * self.max_force
        return steer

    def update(self, dt: float) -> None:
        """Advance one tick: apply steering toward target, integrate."""
        if self.target is not None:
            force = self.seek(self.target)
            self.vel += force * dt

        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        if speed > 1e-6:
            self.heading = math.atan2(self.vel[1], self.vel[0])

        self.pos += self.vel * dt

        # arrive: stop when close to target
        if self.target is not None:
            if np.linalg.norm(self.target - self.pos) < 4.0:
                self.target = None
                self.vel *= 0.0
                self.state = AgentState.IDLE

    @property
    def colour(self) -> tuple[int, int, int]:
        return self.STATE_COLOURS[self.state]
