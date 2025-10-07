import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

@dataclass
class JobInstance:
    r: np.ndarray  # release times
    p: np.ndarray  # processing times
    d: np.ndarray  # due dates

def sample_instance(n=10, r_low=0, r_high=30, p_low=0, p_high=5, extra_due_high=20, seed=None) -> JobInstance:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    r = np.random.uniform(r_low, r_high, size=n).round(0).astype(np.float32)
    p = np.random.uniform(p_low, p_high, size=n).round(0).astype(np.float32)
    slack = np.random.uniform(0, extra_due_high, size=n).astype(np.float32)
    d = (r + p + slack).astype(np.float32)
    return JobInstance(r=r, p=p, d=d)

class OrderEnv:
    """订单调度环境"""
    def __init__(self, inst: JobInstance):
        self.inst = inst
        self.n = len(inst.r)
        self.reset()

    def reset(self):
        self.scheduled = []      # 已调度订单序号
        self.unscheduled = list(range(self.n))
        self.cur_time = 0
        return self._get_state()

    def _get_state(self):
        # 状态：当前时间 + 未调度订单的(r,p,d)
        # 已调度的用mask标记
        mask = np.zeros(self.n, dtype=np.float32)
        for idx in self.scheduled:
            mask[idx] = 1.0
        state = np.concatenate([
            [self.cur_time],
            mask,
            self.inst.r,
            self.inst.p,
            self.inst.d
        ])
        return state

    def step(self, action):
        idx = self.unscheduled[action]
        # 订单释放时间
        self.cur_time = max(self.cur_time, self.inst.r[idx])
        finish_time = self.cur_time + self.inst.p[idx]
        delay = max(0, finish_time - self.inst.d[idx])
        self.scheduled.append(idx)
        self.unscheduled.remove(idx)
        self.cur_time = finish_time
        done = (len(self.unscheduled) == 0)
        next_state = self._get_state()
        reward = -delay  # 总延迟越小越好
        return next_state, reward, done

    def get_total_delay(self, seq):
        cur_time = 0
        total_delay = 0
        for idx in seq:
            cur_time = max(cur_time, self.inst.r[idx])
            finish_time = cur_time + self.inst.p[idx]
            total_delay += max(0, finish_time - self.inst.d[idx])
            cur_time = finish_time
        return total_delay

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, n_orders):
        self.state_dim = 1 + n_orders * 4  # cur_time + mask + r + p + d
        self.action_dim = n_orders
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = []
        self.gamma = 0.99
        self.batch_size = 32
        self.update_steps = 100
        self.learn_step = 0

    def select_action(self, state, valid_actions, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.policy_net(state_tensor).detach().numpy()[0]
        # 只选择未调度订单
        valid_q = [(a, q_values[a]) for a in valid_actions]
        best_action = max(valid_q, key=lambda x: x[1])[0]
        return best_action

    def store(self, state, action, reward, next_state, done, valid_actions):
        self.memory.append((state, action, reward, next_state, done, valid_actions))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, valid_actions_list = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_net(next_states)
        next_q_value = []
        for i, valid_actions in enumerate(valid_actions_list):
            if len(valid_actions) == 0:
                next_q_value.append(0.0)
            else:
                next_q_value.append(next_q_values[i][valid_actions].max().item())
        next_q_value = torch.tensor(next_q_value, dtype=torch.float32)

        expected_q = rewards + self.gamma * next_q_value * (1 - dones)
        loss = nn.MSELoss()(q_value, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def simulated_annealing(env: OrderEnv, seq, max_iter=1000, T0=100, alpha=0.99):
    """对DQN输出的序列用SA优化"""
    best_seq = seq[:]
    best_delay = env.get_total_delay(best_seq)
    cur_seq = seq[:]
    cur_delay = best_delay
    T = T0
    for _ in range(max_iter):
        # 随机交换两个订单
        i, j = random.sample(range(len(seq)), 2)
        new_seq = cur_seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_delay = env.get_total_delay(new_seq)
        delta = new_delay - cur_delay
        if delta < 0 or np.exp(-delta / T) > random.random():
            cur_seq = new_seq
            cur_delay = new_delay
            if cur_delay < best_delay:
                best_seq = cur_seq[:]
                best_delay = cur_delay
        T *= alpha
    return best_seq, best_delay

def main():
    inst = sample_instance(n=10, seed=2)
    env = OrderEnv(inst)
    agent = DQNAgent(n_orders=inst.r.shape[0])

    n_episodes = 200
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        valid_actions = list(range(len(env.unscheduled)))
        while not done:
            # epsilon衰减
            epsilon = max(0.01, 0.1 - episode / n_episodes * 0.09)
            action = agent.select_action(state, valid_actions, epsilon)
            next_state, reward, done = env.step(action)
            next_valid_actions = list(range(len(env.unscheduled)))
            agent.store(state, action, reward, next_state, done, next_valid_actions)
            agent.train()
            state = next_state
            valid_actions = next_valid_actions

    # 测试：用DQN贪心策略生成调度序列
    state = env.reset()
    valid_actions = list(range(len(env.unscheduled)))
    seq = []
    while valid_actions:
        action = agent.select_action(state, valid_actions, epsilon=0.0)
        idx = env.unscheduled[action]
        seq.append(idx)
        _, _, done = env.step(action)
        state = env._get_state()
        valid_actions = list(range(len(env.unscheduled)))
    dqn_delay = env.get_total_delay(seq)
    print("DQN调度序列: ", seq)
    print("DQN总延迟: ", dqn_delay)

    # 用SA优化
    best_seq, best_delay = simulated_annealing(env, seq)
    print("SA优化后调度序列: ", best_seq)
    print("SA优化后总延迟: ", best_delay)

if __name__ == "__main__":
    main()