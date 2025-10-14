# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable, Tuple, List
from tqdm import tqdm

# 统一接口：optimize(eval_fn, n_dim, iters, pop_size, seed)
# 约定：eval_fn(x: np.ndarray, rng: np.random.Generator) -> float，返回“要最小化”的值
# 所有算法都输出：(best_value, best_x, curve)，curve 为每代 best 的轨迹

# ---------------- GWO：灰狼优化 ----------------
def gwo_optimize(eval_fn: Callable[[np.ndarray, np.random.Generator], float],
                 n_dim: int, iters: int, pop_size: int, seed: int = 12
                 ) -> Tuple[float, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((pop_size, n_dim))
    fit = np.array([eval_fn(X[i], rng) for i in tqdm(range(pop_size), desc="GWO init", ncols=0)])

    def top3():
        idx = np.argsort(fit)
        a, b = idx[0], idx[1]
        d = idx[2] if len(idx) > 2 else b
        return int(a), int(b), int(d)

    a_idx, b_idx, d_idx = top3()
    best = float(fit[a_idx]); best_x = X[a_idx].copy()
    curve = [best]

    bar = tqdm(range(1, iters+1), desc="GWO", ncols=0)
    for t in bar:
        a = 2 - 2*(t/iters)
        Xn = np.empty_like(X); fn = np.empty_like(fit)
        for i in range(pop_size):
            r1, r2, r3 = rng.random(n_dim), rng.random(n_dim), rng.random(n_dim)
            A1, C1 = 2*a*r1 - a, 2*rng.random(n_dim)
            A2, C2 = 2*a*r2 - a, 2*rng.random(n_dim)
            A3, C3 = 2*a*r3 - a, 2*rng.random(n_dim)
            X1 = X[a_idx] - A1*np.abs(C1*X[a_idx] - X[i])
            X2 = X[b_idx] - A2*np.abs(C2*X[b_idx] - X[i])
            X3 = X[d_idx] - A3*np.abs(C3*X[d_idx] - X[i])
            Xi = (X1 + X2 + X3)/3.0
            Xn[i] = Xi
            fn[i] = eval_fn(Xn[i], rng)

        worst = int(np.argmax(fn))
        Xn[worst] = best_x.copy(); fn[worst] = best

        X, fit = Xn, fn
        a_idx, b_idx, d_idx = top3()
        if fit[a_idx] < best - 1e-12:
            best = float(fit[a_idx]); best_x = X[a_idx].copy()
        curve.append(best)
        bar.set_postfix_str(f"best={best:.3f}")
    return best, best_x, curve


# ---------------- PSO：粒子群 ----------------
def pso_optimize(eval_fn: Callable[[np.ndarray, np.random.Generator], float],
                 n_dim: int, iters: int, pop_size: int, seed: int = 12,
                 w: float = 0.72, c1: float = 1.49, c2: float = 1.49
                 ) -> Tuple[float, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((pop_size, n_dim))
    V = rng.standard_normal((pop_size, n_dim)) * 0.1
    fit = np.array([eval_fn(X[i], rng) for i in tqdm(range(pop_size), desc="PSO init", ncols=0)])

    pbest = X.copy(); pbest_fit = fit.copy()
    g = int(np.argmin(fit)); gbest = X[g].copy(); gbest_fit = float(fit[g])
    curve = [gbest_fit]

    bar = tqdm(range(1, iters+1), desc="PSO", ncols=0)
    for _ in bar:
        r1 = rng.random((pop_size, n_dim))
        r2 = rng.random((pop_size, n_dim))
        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
        X = X + V
        fit = np.array([eval_fn(X[i], rng) for i in range(pop_size)])

        improved = fit < pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fit[improved]

        g = int(np.argmin(fit))
        if fit[g] < gbest_fit - 1e-12:
            gbest_fit = float(fit[g]); gbest = X[g].copy()

        curve.append(gbest_fit)
        bar.set_postfix_str(f"best={gbest_fit:.3f}")
    return gbest_fit, gbest, curve


# ---------------- VNS：可变邻域搜索（简单实现） ----------------
def vns_optimize(eval_fn: Callable[[np.ndarray, np.random.Generator], float],
                 n_dim: int, iters: int, pop_size: int, seed: int = 12
                 ) -> Tuple[float, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)

    def local_search(x):
        fx = eval_fn(x, rng)
        for k in [1.0, 0.5, 0.2, 0.1]:
            for _ in range(10):
                y = x + rng.standard_normal(x.shape)*k
                # 少量维度随机重置
                mask = rng.random(x.shape) < 0.05
                if mask.any():
                    y[mask] = rng.standard_normal(np.count_nonzero(mask))
                fy = eval_fn(y, rng)
                if fy < fx - 1e-12:
                    x, fx = y, fy
        return fx, x

    X = [rng.standard_normal(n_dim) for _ in range(pop_size)]
    F = [eval_fn(x, rng) for x in tqdm(X, desc="VNS init", ncols=0)]
    g = int(np.argmin(F)); gbest, gfit = X[g].copy(), float(F[g])
    curve = [gfit]

    bar = tqdm(range(1, iters+1), desc="VNS", ncols=0)
    for _ in bar:
        s = int(rng.integers(0, pop_size))
        f2, x2 = local_search(X[s])
        X[s], F[s] = x2, f2
        if f2 < gfit - 1e-12:
            gfit, gbest = f2, x2.copy()
        curve.append(gfit)
        bar.set_postfix_str(f"best={gfit:.3f}")
    return gfit, gbest, curve


# ---------------- AO：Aquila Optimizer（天鹰算法，简实现） ----------------
def ao_optimize(eval_fn: Callable[[np.ndarray, np.random.Generator], float],
                n_dim: int, iters: int, pop_size: int, seed: int = 12
                ) -> Tuple[float, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((pop_size, n_dim))
    fit = np.array([eval_fn(X[i], rng) for i in tqdm(range(pop_size), desc="AO init", ncols=0)])
    g = int(np.argmin(fit)); gbest = X[g].copy(); gbest_fit = float(fit[g])
    curve = [gbest_fit]

    bar = tqdm(range(1, iters+1), desc="AO", ncols=0)
    for t in bar:
        # AO 常见四种策略的简化混合（探索/开采）
        e = 1 - (t/iters)  # 收缩因子
        Xn = np.empty_like(X); fn = np.empty_like(fit)
        for i in range(pop_size):
            r1, r2 = rng.random(n_dim), rng.random(n_dim)
            if rng.random() < 0.5:
                # Exploration：广域搜索（带全局最优牵引）
                A = rng.normal(0, 1, n_dim)
                Xi = gbest + e*(r1*(gbest - A*X[i]))
            else:
                # Exploitation：局部搜索（随机个体 + Levy/高斯）
                j = int(rng.integers(0, pop_size))
                step = rng.normal(0, 1, n_dim) * e
                Xi = X[j] + step*(gbest - X[i])
            Xn[i] = Xi
            fn[i] = eval_fn(Xn[i], rng)

        # 精英保留
        worst = int(np.argmax(fn))
        Xn[worst] = gbest.copy(); fn[worst] = gbest_fit

        X, fit = Xn, fn
        g = int(np.argmin(fit))
        if fit[g] < gbest_fit - 1e-12:
            gbest_fit = float(fit[g]); gbest = X[g].copy()

        curve.append(gbest_fit)
        bar.set_postfix_str(f"best={gbest_fit:.3f}")
    return gbest_fit, gbest, curve


# ---------------- WOA：Whale Optimization Algorithm（鲸鱼算法，简实现） ----------------
def woa_optimize(eval_fn: Callable[[np.ndarray, np.random.Generator], float],
                 n_dim: int, iters: int, pop_size: int, seed: int = 12
                 ) -> Tuple[float, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((pop_size, n_dim))
    fit = np.array([eval_fn(X[i], rng) for i in tqdm(range(pop_size), desc="WOA init", ncols=0)])
    g = int(np.argmin(fit)); gbest = X[g].copy(); gbest_fit = float(fit[g])
    curve = [gbest_fit]

    bar = tqdm(range(1, iters+1), desc="WOA", ncols=0)
    for t in bar:
        a = 2 - 2*(t/iters)  # 线性递减
        Xn = np.empty_like(X); fn = np.empty_like(fit)
        for i in range(pop_size):
            p = rng.random()
            A = 2*a*rng.random(n_dim) - a
            C = 2*rng.random(n_dim)
            if p < 0.5:
                if np.linalg.norm(A, ord=2) < 1:
                    # encircling
                    D = np.abs(C*gbest - X[i])
                    Xi = gbest - A*D
                else:
                    # search random prey
                    j = int(rng.integers(0, pop_size))
                    D = np.abs(C*X[j] - X[i])
                    Xi = X[j] - A*D
            else:
                # spiral updating
                b = 1.0
                l = (rng.random(n_dim) - 0.5)*2  # in [-1,1]
                Dp = np.abs(gbest - X[i])
                Xi = Dp*np.exp(b*l)*np.cos(2*np.pi*l) + gbest
            Xn[i] = Xi
            fn[i] = eval_fn(Xn[i], rng)

        # 精英保留
        worst = int(np.argmax(fn))
        Xn[worst] = gbest.copy(); fn[worst] = gbest_fit

        X, fit = Xn, fn
        g = int(np.argmin(fit))
        if fit[g] < gbest_fit - 1e-12:
            gbest_fit = float(fit[g]); gbest = X[g].copy()

        curve.append(gbest_fit)
        bar.set_postfix_str(f"best={gbest_fit:.3f}")
    return gbest_fit, gbest, curve


