# Entry point. Runs the simulation loop, ticks the coordinator and renderer each frame.

from __future__ import annotations

import sys

import numpy as np
import pygame

from world import World
from renderer import Renderer
from agent import AgentState
from coordinator import assign_zones


FPS = 60


def main() -> None:
    pygame.init()
    world = World.create(n_agents=6)
    renderer = Renderer(world)
    clock = pygame.time.Clock()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # ── events ─────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    world.paused = not world.paused
                elif event.key == pygame.K_r:
                    # scatter agents to random targets
                    for agent in world.agents:
                        agent.target = np.array([
                            np.random.uniform(40, world.width - 40),
                            np.random.uniform(40, world.height - 40),
                        ])
                        agent.state = AgentState.MOVING
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # click to divide the world into Voronoi zones biased toward the cursor
                mx, my = event.pos
                assign_zones(world, np.array([float(mx), float(my)]))

        # ── update & draw ──────────────────────────────────────
        world.step(dt)
        renderer.draw()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
