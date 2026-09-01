"""
CNW: Compositional Navigation World

Environment for FBA-0. A procedurally generated 7x7 grid world with:
- 3 landmarks (A, B, C) at random positions
- Procedural walls (20-30% fill)
- Partial observability (3x3 window, configurable noise)
- Opaque action tokens (ALL 5 actions permuted per world, configurable)
- Configurable observation encoding permutation
- Guaranteed connectivity (BFS check; regenerate on failure)
- One reward per goal (no repeated visits)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


CELL_EMPTY = 0
CELL_WALL = 1
CELL_LANDMARK_A = 2
CELL_LANDMARK_B = 3
CELL_LANDMARK_C = 4
CELL_AGENT = 5
N_CELL_TYPES = 6

ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}
N_ACTIONS = 5

OBS_RADIUS = 1
OBS_SIZE = 2 * OBS_RADIUS + 1
OBS_NOISE = 0.3


@dataclass
class CNWConfig:
    grid_size: int = 7
    n_landmarks: int = 3
    wall_frac_min: float = 0.20
    wall_frac_max: float = 0.30
    obs_noise: float = 0.30
    obs_perm_enabled: bool = True
    action_perm_enabled: bool = True
    max_steps: int = 100
    step_penalty: float = -0.01
    goal_reward: float = 1.0


@dataclass
class CNWWorld:
    grid: np.ndarray
    landmark_positions: list
    action_perm: np.ndarray
    obs_perm: np.ndarray
    config: CNWConfig


@dataclass
class CNWState:
    world: CNWWorld
    agent_pos: tuple
    step: int = 0
    done: bool = False
    goals_reached: set = field(default_factory=set)
    current_goal_idx: int = 0


def _bfs_reachable(grid, start):
    size = grid.shape[0]
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                if grid[nr, nc] != CELL_WALL:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def generate_world(config: CNWConfig, rng: np.random.Generator) -> CNWWorld:
    for _ in range(100):
        world = _try_generate(config, rng)
        if world is not None:
            return world
    raise RuntimeError("Failed to generate connected world after 100 attempts")


def _try_generate(config: CNWConfig, rng: np.random.Generator):
    size = config.grid_size
    grid = np.zeros((size, size), dtype=np.int32)

    n_cells = size * size
    wall_frac = rng.uniform(config.wall_frac_min, config.wall_frac_max)
    n_walls = int(n_cells * wall_frac)

    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    rng.shuffle(interior)
    for r, c in interior[:n_walls]:
        grid[r, c] = CELL_WALL

    empty_cells = [(r, c) for r in range(size) for c in range(size)
                   if grid[r, c] == CELL_EMPTY]
    rng.shuffle(empty_cells)

    n_needed = config.n_landmarks + 1
    if len(empty_cells) < n_needed:
        return None

    landmark_positions = []
    landmark_types = [CELL_LANDMARK_A, CELL_LANDMARK_B, CELL_LANDMARK_C]
    for i in range(config.n_landmarks):
        pos = empty_cells.pop()
        grid[pos] = landmark_types[i]
        landmark_positions.append(pos)

    reachable = _bfs_reachable(grid, landmark_positions[0])
    for pos in landmark_positions[1:]:
        if pos not in reachable:
            return None

    non_wall = [(r, c) for r in range(size) for c in range(size)
                if grid[r, c] != CELL_WALL]
    if not all(pos in reachable for pos in non_wall if pos not in landmark_positions):
        valid_starts = [pos for pos in non_wall if pos in reachable
                        and pos not in landmark_positions]
        if len(valid_starts) < 5:
            return None

    if config.action_perm_enabled:
        action_perm = rng.permutation(N_ACTIONS).astype(np.int32)
    else:
        action_perm = np.arange(N_ACTIONS, dtype=np.int32)

    if config.obs_perm_enabled:
        obs_perm = rng.permutation(N_CELL_TYPES).astype(np.int32)
    else:
        obs_perm = np.arange(N_CELL_TYPES, dtype=np.int32)

    return CNWWorld(
        grid=grid,
        landmark_positions=landmark_positions,
        action_perm=action_perm,
        obs_perm=obs_perm,
        config=config,
    )


def reset(world: CNWWorld, rng: np.random.Generator,
          goal_sequence: Optional[list] = None) -> CNWState:
    reachable = _bfs_reachable(world.grid, world.landmark_positions[0])
    empty_reachable = [pos for pos in reachable
                       if world.grid[pos] == CELL_EMPTY]
    if not empty_reachable:
        empty_reachable = [pos for pos in reachable
                           if world.grid[pos] != CELL_WALL]
    agent_pos = empty_reachable[rng.integers(len(empty_reachable))]
    return CNWState(world=world, agent_pos=agent_pos, step=0,
                    current_goal_idx=0, goals_reached=set())


def get_observation(state: CNWState, rng: np.random.Generator) -> np.ndarray:
    world = state.world
    size = world.config.grid_size
    r, c = state.agent_pos
    obs = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.int32)

    for dr in range(-OBS_RADIUS, OBS_RADIUS + 1):
        for dc in range(-OBS_RADIUS, OBS_RADIUS + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                cell = world.grid[nr, nc]
                if nr == r and nc == c:
                    cell = CELL_AGENT
            else:
                cell = CELL_WALL

            if rng.random() < world.config.obs_noise and not (dr == 0 and dc == 0):
                cell = rng.integers(N_CELL_TYPES)

            obs[dr + OBS_RADIUS, dc + OBS_RADIUS] = world.obs_perm[cell]

    return obs


def obs_to_onehot(obs: np.ndarray) -> np.ndarray:
    flat = obs.flatten()
    onehot = np.zeros((len(flat), N_CELL_TYPES), dtype=np.float32)
    onehot[np.arange(len(flat)), flat] = 1.0
    return onehot.flatten()


def step(state: CNWState, action: int, rng: np.random.Generator,
         goal_idx: Optional[int] = None) -> tuple:
    if state.done:
        obs = get_observation(state, rng)
        return obs_to_onehot(obs), 0.0, True, {"delta_class": 4}

    world = state.world
    true_action = world.action_perm[action]
    dr, dc = ACTIONS[true_action]

    old_pos = state.agent_pos
    r, c = old_pos
    nr, nc = r + dr, c + dc

    size = world.config.grid_size
    if 0 <= nr < size and 0 <= nc < size and world.grid[nr, nc] != CELL_WALL:
        state.agent_pos = (nr, nc)

    actual_dr = state.agent_pos[0] - old_pos[0]
    actual_dc = state.agent_pos[1] - old_pos[1]
    delta_class = _delta_to_class(actual_dr, actual_dc)

    state.step += 1

    target_idx = goal_idx if goal_idx is not None else state.current_goal_idx
    reward = world.config.step_penalty
    done = False

    if (state.agent_pos == world.landmark_positions[target_idx]
            and target_idx not in state.goals_reached):
        reward = world.config.goal_reward
        state.goals_reached.add(target_idx)
        state.current_goal_idx += 1
        if state.current_goal_idx >= world.config.n_landmarks:
            done = True

    if state.step >= world.config.max_steps:
        done = True

    state.done = done
    obs = get_observation(state, rng)
    info = {"goals_reached": list(state.goals_reached), "delta_class": delta_class}
    return obs_to_onehot(obs), reward, done, info


def _delta_to_class(dr, dc):
    if (dr, dc) == (0, -1): return 0
    if (dr, dc) == (0, 1): return 1
    if (dr, dc) == (-1, 0): return 2
    if (dr, dc) == (1, 0): return 3
    return 4


def obs_dim():
    return OBS_SIZE * OBS_SIZE * N_CELL_TYPES
