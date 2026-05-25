# Classical coordination layer: Voronoi area partitioning, APF collision avoidance, consensus sync.

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

from agent import AgentState
from world import World


# Fraction of pull toward the click point when building the navigation target:
# target = (1 - CLICK_BIAS) * centroid + CLICK_BIAS * click.
# Higher = agents pile toward the click, lower = agents stay spread in their own zone.
CLICK_BIAS = 0.4


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area-weighted centroid of a 2D polygon (vertices in order)."""
    x = poly[:, 0]
    y = poly[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def _isect_x(a: np.ndarray, b: np.ndarray, x: float) -> np.ndarray:
    """Intersection of segment a→b with vertical line X=x."""
    dx = b[0] - a[0]
    if abs(dx) < 1e-12:
        return np.array([x, a[1]])
    t = (x - a[0]) / dx
    return np.array([x, a[1] + t * (b[1] - a[1])])


def _isect_y(a: np.ndarray, b: np.ndarray, y: float) -> np.ndarray:
    """Intersection of segment a→b with horizontal line Y=y."""
    dy = b[1] - a[1]
    if abs(dy) < 1e-12:
        return np.array([a[0], y])
    t = (y - a[1]) / dy
    return np.array([a[0] + t * (b[0] - a[0]), y])


def clip_polygon_to_rect(poly: np.ndarray, x_min: float, y_min: float,
                         x_max: float, y_max: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon against an axis-aligned rect."""
    edges = (
        (lambda p: p[0] >= x_min, lambda a, b: _isect_x(a, b, x_min)),
        (lambda p: p[0] <= x_max, lambda a, b: _isect_x(a, b, x_max)),
        (lambda p: p[1] >= y_min, lambda a, b: _isect_y(a, b, y_min)),
        (lambda p: p[1] <= y_max, lambda a, b: _isect_y(a, b, y_max)),
    )
    output = [np.asarray(p, dtype=float) for p in poly]
    for inside, intersect in edges:
        if not output:
            break
        input_list = output
        output = []
        prev = input_list[-1]
        for curr in input_list:
            if inside(curr):
                if not inside(prev):
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif inside(prev):
                output.append(intersect(prev, curr))
            prev = curr
    return np.array(output) if output else np.empty((0, 2))


def _pull_inside_polygon(point: np.ndarray, polygon: np.ndarray,
                         anchor: np.ndarray, margin: float = 0.95) -> np.ndarray:
    """If *point* lies outside the convex *polygon*, pull it back along the
    *anchor*→*point* segment to the first edge crossing, then scale by *margin*
    to stay strictly inside. *anchor* must be inside the polygon."""
    a = np.asarray(anchor, dtype=float)
    p = np.asarray(point, dtype=float)
    t_exit = 1.0
    n = len(polygon)
    for i in range(n):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % n]
        ex, ey = v2[0] - v1[0], v2[1] - v1[1]
        # signed area of edge with anchor / point — same sign means same side
        s_a = ex * (a[1] - v1[1]) - ey * (a[0] - v1[0])
        s_p = ex * (p[1] - v1[1]) - ey * (p[0] - v1[0])
        if s_a * s_p < 0:  # segment crosses this edge
            t = s_a / (s_a - s_p)
            if t < t_exit:
                t_exit = t
    if t_exit < 1.0:
        return a + (t_exit * margin) * (p - a)
    return p


def assign_zones(world: World, click: np.ndarray) -> None:
    """Divide the world into Voronoi zones from current agent positions and
    assign each agent a target blended between its zone centroid and *click*
    using CLICK_BIAS.

    Cells are bounded by mirror-reflecting each agent across the four world
    borders and then explicitly clipped to the world rectangle as a safety
    net against degenerate or infinite regions.
    """
    if not world.agents:
        return
    click = np.asarray(click, dtype=float)
    positions = np.array([a.pos for a in world.agents])
    w, h = world.width, world.height
    mirrored = np.vstack([
        positions,
        np.column_stack([-positions[:, 0],         positions[:, 1]]),
        np.column_stack([2 * w - positions[:, 0],  positions[:, 1]]),
        np.column_stack([positions[:, 0],          -positions[:, 1]]),
        np.column_stack([positions[:, 0],           2 * h - positions[:, 1]]),
    ])
    vor = Voronoi(mirrored)
    for i, agent in enumerate(world.agents):
        region = vor.regions[vor.point_region[i]]
        # skip empty or unbounded regions — defensive; mirror trick should prevent these
        if not region or -1 in region:
            continue
        polygon = clip_polygon_to_rect(
            vor.vertices[region], 0.0, 0.0, float(w), float(h),
        )
        if len(polygon) < 3:
            continue
        agent.zone = polygon
        centroid = _polygon_centroid(polygon)
        raw_target = (1.0 - CLICK_BIAS) * centroid + CLICK_BIAS * click
        agent.target = _pull_inside_polygon(raw_target, polygon, centroid)
        agent.state = AgentState.MOVING
