# Pygame rendering: draws agents, zones, coverage heatmap, threats, and HUD each frame.

from __future__ import annotations

import math

import pygame

from agent import Agent
from world import World
from coordinator import clip_polygon_to_rect


class Renderer:
    """Pygame-ce renderer for the swarm simulation."""

    BG_COLOUR = (15, 15, 25)
    GRID_COLOUR = (30, 30, 45)
    HUD_COLOUR = (200, 200, 200)
    ZONE_COLOUR = (70, 90, 120)

    def __init__(self, world: World) -> None:
        self.world = world
        self.screen = pygame.display.set_mode(
            (world.width, world.height),
        )
        pygame.display.set_caption("Swarm Simulator")
        self.font = pygame.font.SysFont("consolas", 14)

    # ── public ─────────────────────────────────────────────────────
    def draw(self) -> None:
        self.screen.fill(self.BG_COLOUR)
        self._draw_grid()
        self._draw_zones()
        for agent in self.world.agents:
            self._draw_agent(agent)
        self._draw_hud()
        pygame.display.flip()

    # ── internals ──────────────────────────────────────────────────
    def _draw_zones(self) -> None:
        """Outline each agent's assigned Voronoi zone, clipped to the screen rect."""
        w, h = float(self.world.width), float(self.world.height)
        for agent in self.world.agents:
            if agent.zone is None or len(agent.zone) < 3:
                continue
            clipped = clip_polygon_to_rect(agent.zone, 0.0, 0.0, w, h)
            if len(clipped) < 3:
                continue
            points = [(int(p[0]), int(p[1])) for p in clipped]
            pygame.draw.polygon(self.screen, self.ZONE_COLOUR, points, 1)

    def _draw_grid(self, spacing: int = 60) -> None:
        for x in range(0, self.world.width, spacing):
            pygame.draw.line(self.screen, self.GRID_COLOUR, (x, 0), (x, self.world.height))
        for y in range(0, self.world.height, spacing):
            pygame.draw.line(self.screen, self.GRID_COLOUR, (0, y), (self.world.width, y))

    def _draw_agent(self, agent: Agent) -> None:
        """Draw agent as a pointed triangle showing heading."""
        r = agent.radius
        h = agent.heading
        # triangle vertices: nose, left wing, right wing
        nose = (
            agent.pos[0] + math.cos(h) * r * 1.6,
            agent.pos[1] + math.sin(h) * r * 1.6,
        )
        left = (
            agent.pos[0] + math.cos(h + 2.5) * r,
            agent.pos[1] + math.sin(h + 2.5) * r,
        )
        right = (
            agent.pos[0] + math.cos(h - 2.5) * r,
            agent.pos[1] + math.sin(h - 2.5) * r,
        )
        pygame.draw.polygon(self.screen, agent.colour, [nose, left, right])

        # id label
        label = self.font.render(str(agent.id), True, (180, 180, 180))
        self.screen.blit(label, (agent.pos[0] + r + 3, agent.pos[1] - 6))

    def _draw_hud(self) -> None:
        lines = [
            f"Agents: {len(self.world.agents)}",
            f"Tick:   {self.world.tick}",
            f"{'PAUSED' if self.world.paused else 'RUNNING'}",
        ]
        y = 8
        for line in lines:
            surf = self.font.render(line, True, self.HUD_COLOUR)
            self.screen.blit(surf, (10, y))
            y += 18
