"""DQN + Simulated Annealing hybrid for single machine total tardiness minimisation.

This module defines a reinforcement learning environment for sequencing jobs with
release dates, processing times and due dates on a single machine.  A Deep Q
Network (DQN) agent is trained to produce an initial sequence, which is then
refined using a simulated annealing (SA) local search.

The code is self-contained and can be executed directly.  When run as a script
it will sample a random instance, train the agent for a small number of
episodes, apply the SA improvement and finally print the resulting schedule
along with the associated tardiness values.

The implementation purposefully favours clarity over raw performance so that it
can serve as a starting point for further experimentation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Problem definition utilities


@dataclass
class JobInstance:
    """Container for a single machine scheduling instance."""

    r: np.ndarray  # release times
    p: np.ndarray  # processing times
    d: np.ndarray  # due dates


def sample_instance(
    n: int = 10,
    r_low: float = 0,
    r_high: float = 30,
    p_low: float = 1,
    p_high: float = 5,
    extra_due_high: float = 20,
    seed: int | None = None,
) -> JobInstance:
    """Sample a random scheduling instance.

    Parameters
    ----------
    n:
        Number of jobs to generate.
    r_low, r_high:
        Range for job release times.
    p_low, p_high:
        Range for processing times.  ``p_low`` is set to 1 by default to avoid
        zero-length jobs that can lead to degenerate rewards.
    extra_due_high:
        Upper bound for the additional slack added to compute the due dates.
    seed:
        Random seed for reproducibility.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    r = np.random.uniform(r_low, r_high, size=n).round(0).astype(np.float32)
    p = np.random.uniform(p_low, p_high, size=n).round(0).astype(np.float32)
    slack = np.random.uniform(0, extra_due_high, size=n).astype(np.float32)
    d = (r + p + slack).astype(np.float32)
    return JobInstance(r=r, p=p, d=d)


def compute_schedule_tardiness(order: Sequence[int], instance: JobInstance) -> float:
    """Return the total tardiness of ``order`` for ``instance``."""

    time = 0.0
    total_tardiness = 0.0
    for job in order:
        start = max(time, float(instance.r[job]))
        completion = start + float(instance.p[job])
        tardiness = max(0.0, completion - float(instance.d[job]))
        total_tardiness += tardiness
        time = completion
    return total_tardiness


# ---------------------------------------------------------------------------
# Environment definition


class JobSchedulingEnv:
    """Environment for the DQN agent.

    The state is represented as a flattened vector containing job features and
    an extra component with the current time.  Each job contributes:

    - Release time (normalised)
    - Processing time (normalised)
    - Due date (normalised)
    - Completion flag (1 if already scheduled, else 0)
    - Availability flag (1 if release time <= current time, else 0)
    """

    invalid_action_penalty: float = -5.0

    def __init__(self, instance: JobInstance):
        self.instance = instance
        self.n_jobs = len(instance.r)
        self.max_time = float(np.max(instance.d) + np.max(instance.p))
        self.state: np.ndarray | None = None
        self.current_time: float = 0.0
        self.done_mask: np.ndarray | None = None
        self.total_tardiness: float = 0.0

    def reset(self) -> np.ndarray:
        self.current_time = 0.0
        self.done_mask = np.zeros(self.n_jobs, dtype=bool)
        self.total_tardiness = 0.0
        self.state = self._build_state()
        return self.state.copy()

    def _build_state(self) -> np.ndarray:
        assert self.done_mask is not None
        features = []
        for idx in range(self.n_jobs):
            r = float(self.instance.r[idx]) / self.max_time
            p = float(self.instance.p[idx]) / self.max_time
            d = float(self.instance.d[idx]) / self.max_time
            done = 1.0 if self.done_mask[idx] else 0.0
            available = 1.0 if float(self.instance.r[idx]) <= self.current_time else 0.0
            features.extend([r, p, d, done, available])
        # Append the normalised current time as a global feature.
        features.append(self.current_time / self.max_time)
        return np.array(features, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert self.done_mask is not None

        reward = 0.0
        done = False

        if self.done_mask[action]:
            # Selecting an already scheduled job is penalised to encourage
            # exploring feasible sequences.
            reward = self.invalid_action_penalty
        else:
            start = max(self.current_time, float(self.instance.r[action]))
            completion = start + float(self.instance.p[action])
            tardiness = max(0.0, completion - float(self.instance.d[action]))

            reward = -tardiness
            self.total_tardiness += tardiness
            self.current_time = completion
            self.done_mask[action] = True

            if self.done_mask.all():
                done = True

        self.state = self._build_state()
        return self.state.copy(), reward, done


# ---------------------------------------------------------------------------
# Deep Q-Network agent


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(
    policy_net: DQN,
    state: np.ndarray,
    epsilon: float,
    n_actions: int,
    scheduled_mask: np.ndarray,
) -> int:
    """Epsilon-greedy action selection over the set of unscheduled jobs."""

    unscheduled = np.where(~scheduled_mask)[0]
    if len(unscheduled) == 0:
        # Fall back to any action to avoid crashes (environment will finish).
        return int(np.random.randint(0, n_actions))

    if random.random() < epsilon:
        return int(random.choice(list(unscheduled)))

    with torch.no_grad():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        q_values = policy_net(state_tensor).squeeze(0).numpy()

    # Mask already scheduled jobs by assigning a large negative value.
    masked_q = q_values.copy()
    masked_q[scheduled_mask] = -1e9
    return int(np.argmax(masked_q))


def train_dqn(
    instance: JobInstance,
    episodes: int = 500,
    gamma: float = 0.95,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update: int = 20,
    replay_capacity: int = 20_000,
) -> Tuple[DQN, List[float]]:
    """Train a DQN agent for the provided instance."""

    env = JobSchedulingEnv(instance)
    state_dim = env.reset().shape[0]
    n_actions = env.n_jobs

    policy_net = DQN(state_dim, n_actions)
    target_net = DQN(state_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimiser = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(replay_capacity)

    epsilon = epsilon_start
    episode_rewards: List[float] = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            scheduled_mask = env.done_mask.copy() if env.done_mask is not None else np.zeros(n_actions, dtype=bool)
            action = select_action(policy_net, state, epsilon, n_actions, scheduled_mask)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ) = replay_buffer.sample(batch_size)

                states_t = torch.from_numpy(states).float()
                actions_t = torch.from_numpy(actions).long().unsqueeze(1)
                rewards_t = torch.from_numpy(rewards).float().unsqueeze(1)
                next_states_t = torch.from_numpy(next_states).float()
                dones_t = torch.from_numpy(dones).float().unsqueeze(1)

                q_values = policy_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1, keepdim=True)[0]
                    expected_q = rewards_t + gamma * next_q * (1.0 - dones_t)

                loss = loss_fn(q_values, expected_q)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        episode_rewards.append(total_reward)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, episode_rewards


def extract_schedule(policy_net: DQN, instance: JobInstance) -> List[int]:
    """Generate a schedule by greedily following the learned policy."""

    env = JobSchedulingEnv(instance)
    state = env.reset()
    order: List[int] = []
    done = False

    while not done:
        done_mask = env.done_mask.copy() if env.done_mask is not None else np.zeros(env.n_jobs, dtype=bool)
        action = select_action(
            policy_net,
            state,
            epsilon=0.0,
            n_actions=env.n_jobs,
            scheduled_mask=done_mask,
        )
        state, _, done = env.step(action)
        if env.done_mask is not None and not done_mask[action] and env.done_mask[action]:
            order.append(action)

    return order


# ---------------------------------------------------------------------------
# Simulated annealing improvement


def simulated_annealing(
    initial_order: Sequence[int],
    instance: JobInstance,
    initial_temperature: float = 10.0,
    cooling_rate: float = 0.95,
    iterations_per_temp: int = 100,
    min_temperature: float = 1e-3,
) -> Tuple[List[int], float]:
    """Improve ``initial_order`` using simulated annealing."""

    current_order = list(initial_order)
    current_value = compute_schedule_tardiness(current_order, instance)
    best_order = list(current_order)
    best_value = current_value

    temperature = initial_temperature

    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            i, j = np.random.choice(len(current_order), size=2, replace=False)
            neighbour = current_order.copy()
            neighbour[i], neighbour[j] = neighbour[j], neighbour[i]

            neighbour_value = compute_schedule_tardiness(neighbour, instance)
            delta = neighbour_value - current_value

            if delta < 0 or math.exp(-delta / temperature) > random.random():
                current_order = neighbour
                current_value = neighbour_value

                if neighbour_value < best_value:
                    best_order = neighbour
                    best_value = neighbour_value

        temperature *= cooling_rate

    return best_order, best_value


# ---------------------------------------------------------------------------
# Convenience runner


def solve_with_dqn_sa(
    instance: JobInstance,
    dqn_episodes: int = 500,
    sa_temperature: float = 10.0,
    sa_cooling: float = 0.9,
    sa_iterations: int = 50,
) -> Tuple[List[int], float]:
    """Train a DQN, refine the solution with SA and return the best schedule."""

    policy_net, rewards = train_dqn(instance, episodes=dqn_episodes)
    base_order = extract_schedule(policy_net, instance)
    improved_order, improved_value = simulated_annealing(
        base_order,
        instance,
        initial_temperature=sa_temperature,
        cooling_rate=sa_cooling,
        iterations_per_temp=sa_iterations,
    )

    print("Training rewards (last 10 episodes):", rewards[-10:])
    print("DQN schedule:", base_order)
    print("DQN total tardiness:", compute_schedule_tardiness(base_order, instance))
    print("Improved schedule:", improved_order)
    print("Improved total tardiness:", improved_value)

    return improved_order, improved_value


if __name__ == "__main__":
    instance = sample_instance(n=10, seed=2)
    print("Release times:", instance.r)
    print("Processing times:", instance.p)
    print("Due dates:", instance.d)

    solve_with_dqn_sa(
        instance,
        dqn_episodes=200,  # fewer episodes for a quick demonstration
        sa_temperature=5.0,
        sa_cooling=0.9,
        sa_iterations=40,
    )
