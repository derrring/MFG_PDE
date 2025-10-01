"""Smoke tests for Mean Field Q-Learning with the maze environment."""

import pytest

import numpy as np

try:
    import gymnasium as gym  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GYMNASIUM_AVAILABLE = False

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False

from mfg_pde.alg.reinforcement.algorithms.mean_field_q_learning import create_mean_field_q_learning
from mfg_pde.alg.reinforcement.environments import (
    ActionType,
    MFGMazeConfig,
    MFGMazeEnvironment,
    RewardType,
)


@pytest.mark.skipif(not (GYMNASIUM_AVAILABLE and TORCH_AVAILABLE), reason="Gymnasium and torch required")
def test_mean_field_q_learning_smoke() -> None:
    """Ensure the Q-learning loop runs for a tiny episode without crashing."""
    maze = np.ones((6, 6), dtype=np.int32)
    maze[1:5, 1:5] = 0  # 4x4 open room

    config = MFGMazeConfig(
        maze_array=maze,
        start_positions=[(1, 1), (1, 4)],
        goal_positions=[(4, 4), (4, 1)],
        action_type=ActionType.FOUR_CONNECTED,
        reward_type=RewardType.DENSE,
        max_episode_steps=5,
        num_agents=2,
    )

    env = MFGMazeEnvironment(config)

    solver = create_mean_field_q_learning(
        env,
        {
            "batch_size": 4,
            "replay_buffer_size": 32,
            "target_update_frequency": 1,
            "epsilon": 0.2,
            "epsilon_decay": 1.0,
        },
    )

    results = solver.train(num_episodes=1)

    assert len(results["episode_rewards"]) == 1
    assert isinstance(results["episode_rewards"][0], float)
