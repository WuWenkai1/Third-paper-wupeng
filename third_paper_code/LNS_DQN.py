# -*- coding: utf-8 -*-
"""LNS+DQN trainer with experience replay and ranking-based objectives."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ===== 读取你已有的模型封装（不改动它）=====
from Main_model import (  # type: ignore
    SCENARIO_AS,
    construct_schedule,
    normal_indices,
    orders,
)

# ===== 全局与超参 =====
SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

T_TARGET = 409_865.500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 800
LNS_ITERS_PER_EP = 20
DESTROY_K_MIN, DESTROY_K_MAX = 6, 20
SA_T0, SA_TMIN = 5.0, 0.1
SA_COOL = 0.85

LR = 1e-3
BATCH = 64
EPS_START, EPS_END, EPS_DECAY = 0.7, 0.05, 0.98
REPLAY = 12_000
WARMUP_EP = 20
UPDATE_STEPS = 6
TAU = 0.01
REWARD_SCALE = 2000.0


def compute_PT_from_info(info: Dict, Z: float = T_TARGET) -> float:
    return float(
        sum(max(0.0, Z - float(info["f_values"][a])) for a in SCENARIO_AS)
    )


def wspt_key(j: int) -> Tuple[float, float]:
    pb = float(np.sum(orders[j]["proc_body"]))
    pc = float(np.sum(orders[j]["proc_cabinet"]))
    pp = float(orders[j]["proc_pipe"])
    p_eff = max(pb, pc) + pp
    w = max(1e-9, float(orders[j]["penalty"]))
    return (p_eff / w, float(orders[j]["due"]))


def rebuild_from_priority(x: np.ndarray, tail_rule: str = "WSPT") -> Tuple[Dict, float]:
    _, _, _, _, info = construct_schedule(priority_normals=x, tail_rule=tail_rule)
    PT = compute_PT_from_info(info)
    return info, PT


def vector_from_insert_order(
    insert_order: Iterable[int], base: float = 10.0, decay: float = 0.1
) -> np.ndarray:
    x = np.zeros((len(normal_indices),), dtype=np.float32)
    pos = {job: i for i, job in enumerate(normal_indices)}
    score = base
    for j in insert_order:
        if j in pos:
            x[pos[j]] = max(score, 0.0)
            score -= decay
    return x


def job_features_from_info(info: Dict) -> np.ndarray:
    feats = []
    for j in normal_indices:
        pb = float(np.sum(orders[j]["proc_body"]))
        pc = float(np.sum(orders[j]["proc_cabinet"]))
        pp = float(orders[j]["proc_pipe"])
        r = float(orders[j]["release"])
        d = float(orders[j]["due"])
        w = float(orders[j]["penalty"])
        pr = float(orders[j]["profit"])
        ac = float(orders[j]["AC"])
        p_eff = max(pb, pc) + pp
        slack = d - r - p_eff
        C_pipe = float(info.get("pipe_finish", {}).get(j, 0.0))
        profit_density = (pr - ac * (pb + pc + pp)) / max(1e-6, p_eff)
        feats.append(
            [pb, pc, pp, p_eff, r, d, w, pr, ac, slack, C_pipe, profit_density]
        )
    return np.asarray(feats, dtype=np.float32)


class NetJobScore(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class DestroyTransition:
    features: np.ndarray
    mask: np.ndarray
    reward: float


@dataclass
class RepairTransition:
    features: np.ndarray
    order: List[int]
    reward: float


class Replay:
    def __init__(self, cap: int = REPLAY):
        self.buf: List[DestroyTransition | RepairTransition] = []
        self.cap = cap
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, item: DestroyTransition | RepairTransition) -> None:
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.cap

    def sample(self, bs: int) -> List[DestroyTransition | RepairTransition]:
        idx = np.random.choice(len(self.buf), size=bs, replace=False)
        return [self.buf[i] for i in idx]


def _tanh_reward(delta_pt: float) -> float:
    return float(np.tanh(delta_pt / REWARD_SCALE))


def initial_solution() -> Tuple[np.ndarray, Dict, float]:
    normals_sorted = sorted(normal_indices, key=wspt_key)
    x0 = vector_from_insert_order(normals_sorted, base=10.0, decay=0.01)
    info0, PT0 = rebuild_from_priority(x0, tail_rule="WSPT")
    return x0, info0, PT0


def destroy_with_policy(
    info: Dict, eps: float, net: NetJobScore, K: int
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    feats = job_features_from_info(info)
    with torch.no_grad():
        scores = net(torch.tensor(feats, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    if np.random.rand() < eps:
        cand = list(range(len(normal_indices)))
        random.shuffle(cand)
        pick_idx = cand[:K]
    else:
        pick_idx = np.argsort(-scores)[:K]
    jobs = [normal_indices[i] for i in pick_idx]
    mask = np.zeros((len(normal_indices),), dtype=np.float32)
    mask[pick_idx] = 1.0
    return jobs, feats, mask


def repair_order_with_policy(
    removed_jobs: List[int], feats_removed: np.ndarray, eps: float, net: NetJobScore
) -> List[int]:
    with torch.no_grad():
        scores = net(torch.tensor(feats_removed, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    if np.random.rand() < eps:
        order = list(range(len(removed_jobs)))
        random.shuffle(order)
    else:
        order = list(np.argsort(-scores))
    return [removed_jobs[i] for i in order]


def lns_one_iter(
    x: np.ndarray,
    net_destroy: NetJobScore,
    net_repair: NetJobScore,
    eps: float,
) -> Tuple[np.ndarray, float, Dict, DestroyTransition, RepairTransition]:
    info, PT = rebuild_from_priority(x, tail_rule="WSPT")

    K_span = max(1, DESTROY_K_MAX - DESTROY_K_MIN)
    q = min(1.0, max(0.0, PT / (PT + 1e3)))
    K = DESTROY_K_MIN + int(K_span * q)

    removed_jobs, feats_before, mask = destroy_with_policy(info, eps, net_destroy, K)

    x_removed = x.copy()
    for j in removed_jobs:
        idx = normal_indices.index(j)
        x_removed[idx] = 0.0

    info_removed, _ = rebuild_from_priority(x_removed)
    feats_removed_all = job_features_from_info(info_removed)
    id2pos = {j: i for i, j in enumerate(normal_indices)}
    removed_idx = [id2pos[j] for j in removed_jobs]
    feats_removed = feats_removed_all[removed_idx]

    order = repair_order_with_policy(removed_jobs, feats_removed, eps, net_repair)

    base, decay = 12.0, 0.2
    x_new = x_removed.copy()
    for pos, j in enumerate(order):
        idx = normal_indices.index(j)
        x_new[idx] = base - pos * decay

    info_new, PT_new = rebuild_from_priority(x_new)

    delta = PT - PT_new
    reward = _tanh_reward(delta)

    destroy_tr = DestroyTransition(features=feats_before, mask=mask, reward=reward)

    orig_positions = {job: idx for idx, job in enumerate(removed_jobs)}
    order_indices = [orig_positions[j] for j in order]
    repair_tr = RepairTransition(
        features=feats_removed,
        order=order_indices,
        reward=reward,
    )

    return x_new, PT_new, info_new, destroy_tr, repair_tr


def _destroy_loss(batch: List[DestroyTransition], net: NetJobScore) -> torch.Tensor:
    if not batch:
        return torch.tensor(0.0, device=DEVICE)

    losses = []
    for tr in batch:
        feats = torch.tensor(tr.features, dtype=torch.float32, device=DEVICE)
        mask = torch.tensor(tr.mask > 0.5, dtype=torch.bool, device=DEVICE)
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        scores = net(feats)
        pos = scores[mask]
        neg = scores[~mask]
        margin = 1.0 - (pos.unsqueeze(1) - neg.unsqueeze(0))
        margin = torch.clamp(margin, min=0.0)
        weight = 1.0 + tr.reward
        losses.append((margin.pow(2).mean()) * weight)
    if not losses:
        return torch.tensor(0.0, device=DEVICE)
    return torch.stack(losses).mean()


def _repair_loss(batch: List[RepairTransition], net: NetJobScore) -> torch.Tensor:
    if not batch:
        return torch.tensor(0.0, device=DEVICE)

    losses = []
    for tr in batch:
        if len(tr.order) <= 1:
            continue
        feats = torch.tensor(tr.features, dtype=torch.float32, device=DEVICE)
        scores = net(feats)
        loss = 0.0
        pair_cnt = 0
        for u in range(len(tr.order)):
            for v in range(u + 1, len(tr.order)):
                su = scores[tr.order[u]]
                sv = scores[tr.order[v]]
                margin = torch.clamp(1.0 - (su - sv), min=0.0)
                loss = loss + margin.pow(2)
                pair_cnt += 1
        if pair_cnt == 0:
            continue
        weight = 1.0 + tr.reward
        losses.append(loss / pair_cnt * weight)
    if not losses:
        return torch.tensor(0.0, device=DEVICE)
    return torch.stack(losses).mean()


def soft_update(target: nn.Module, online: nn.Module, tau: float = TAU) -> None:
    with torch.no_grad():
        for pt, p in zip(target.parameters(), online.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)


def train() -> None:
    x, info, PT = initial_solution()
    best_x, best_PT = x.copy(), float(PT)

    F = job_features_from_info(info).shape[1]
    net_d = NetJobScore(in_dim=F).to(DEVICE)
    net_r = NetJobScore(in_dim=F).to(DEVICE)
    tgt_d = NetJobScore(in_dim=F).to(DEVICE)
    tgt_r = NetJobScore(in_dim=F).to(DEVICE)
    tgt_d.load_state_dict(net_d.state_dict())
    tgt_r.load_state_dict(net_r.state_dict())

    opt_d = optim.Adam(net_d.parameters(), lr=LR)
    opt_r = optim.Adam(net_r.parameters(), lr=LR)

    destroy_replay = Replay(REPLAY)
    repair_replay = Replay(REPLAY)

    eps = EPS_START
    curves_PT, curves_acc, curves_eps = [], [], []

    bar = tqdm(range(1, EPISODES + 1), desc="Train LNS(main)+DQN(aux)", ncols=0)
    for ep in bar:
        T = max(SA_TMIN, SA_T0 * (SA_COOL ** (ep - 1)))
        acc_cnt = 0

        for _ in range(LNS_ITERS_PER_EP):
            x_new, PT_new, info_new, destroy_tr, repair_tr = lns_one_iter(
                x, net_d, net_r, eps
            )

            if PT_new < PT - 1e-9:
                x, PT, info = x_new, PT_new, info_new
                acc_cnt += 1
            else:
                prob = math.exp(-(PT_new - PT) / max(1e-9, T))
                if np.random.rand() < prob:
                    x, PT, info = x_new, PT_new, info_new
                    acc_cnt += 1

            if PT < best_PT - 1e-9:
                best_PT, best_x = PT, x.copy()

            destroy_replay.push(destroy_tr)
            repair_replay.push(repair_tr)

            if ep > WARMUP_EP:
                for _ in range(UPDATE_STEPS):
                    if len(destroy_replay) >= BATCH:
                        batch = destroy_replay.sample(BATCH)
                        loss_d = _destroy_loss(batch, net_d)
                        if loss_d.requires_grad:
                            opt_d.zero_grad()
                            loss_d.backward()
                            nn.utils.clip_grad_norm_(net_d.parameters(), 5.0)
                            opt_d.step()

                    if len(repair_replay) >= BATCH:
                        batch_r = repair_replay.sample(BATCH)
                        loss_r = _repair_loss(batch_r, net_r)
                        if loss_r.requires_grad:
                            opt_r.zero_grad()
                            loss_r.backward()
                            nn.utils.clip_grad_norm_(net_r.parameters(), 5.0)
                            opt_r.step()

                soft_update(tgt_d, net_d)
                soft_update(tgt_r, net_r)

        eps = max(EPS_END, eps * EPS_DECAY)
        curves_PT.append(PT)
        curves_acc.append(acc_cnt)
        curves_eps.append(eps)
        bar.set_postfix_str(
            f"PT={PT:.1f} | best={best_PT:.1f} | acc={acc_cnt} | eps={eps:.2f}"
        )

    plt.figure(figsize=(7, 4))
    plt.plot(curves_PT, label="PT (lower better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lns_dqn_training.png", dpi=150)

    np.save("best_priority_normals.npy", best_x)
    torch.save(
        {
            "destroy": net_d.state_dict(),
            "repair": net_r.state_dict(),
            "PT_curve": curves_PT,
            "acc_curve": curves_acc,
            "eps_curve": curves_eps,
            "best_PT": best_PT,
        },
        "dqn_destroy_repair.pt",
    )
    print(
        "[Saved] dqn_destroy_repair.pt, lns_dqn_training.png, best_priority_normals.npy"
    )
    print(f"[Done] Best PT = {best_PT:.2f}")


if __name__ == "__main__":
    train()

