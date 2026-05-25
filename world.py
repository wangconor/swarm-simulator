# Shared simulation state: terrain grid, threat objects, coverage map, and mission status.

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from agent import Agent, AgentState


@dataclass
class World:
    """Holds all shared simulation state."""

    width: int = 1200
    height: int = 800
    agents: list[Agent] = field(default_factory=list)
    paused: bool = False
    tick: int = 0

    # ── factory ────────────────────────────────────────────────────
    @classmethod
    def create(cls, n_agents: int = 6, **kwargs) -> "World":
        """Spawn a world with *n_agents* agents at random positions."""
        world = cls(**kwargs)
        margin = 80
        for i in range(n_agents):
            pos = np.array([
                random.uniform(margin, world.width - margin),
                random.uniform(margin, world.height - margin),
            ])
            world.agents.append(Agent(id=i, pos=pos))
        return world

    # ── simulation step ────────────────────────────────────────────
    def step(self, dt: float) -> None:
        if self.paused:
            return
        for agent in self.agents:
            agent.update(dt)
        self._resolve_collisions()
        self.tick += 1

    def _resolve_collisions(self) -> None:
        """Push apart any overlapping pairs and cancel inward velocity."""
        agents = self.agents
        for i in range(len(agents)):
            a = agents[i]
            for j in range(i + 1, len(agents)):
                b = agents[j]
                delta = b.pos - a.pos
                dist = float(np.linalg.norm(delta))
                min_dist = a.radius + b.radius
                if dist >= min_dist:
                    continue
                if dist < 1e-6:
                    # exact overlap — pick an arbitrary separation axis
                    delta = np.array([1.0, 0.0])
                    dist = 1.0
                normal = delta / dist
                overlap = min_dist - dist
                a.pos -= normal * (overlap / 2)
                b.pos += normal * (overlap / 2)
                # cancel velocity components that would re-collide
                a_into = float(np.dot(a.vel, normal))
                if a_into > 0:
                    a.vel -= normal * a_into
                b_into = float(np.dot(b.vel, normal))
                if b_into < 0:
                    b.vel -= normal * b_into

    # ── queries ────────────────────────────────────────────────────
    def agent_positions(self) -> np.ndarray:
        """Return (N, 2) array of all agent positions."""
        return np.array([a.pos for a in self.agents])
