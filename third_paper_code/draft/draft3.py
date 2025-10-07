import numpy as np
import random
from typing import Optional, Tuple
from draft import sample_instance, JobInstance



# ---------- 评价函数：给定顺序 -> 计算总延期 ----------
def sequence_total_tardiness(seq: np.ndarray, inst: JobInstance) -> float:
    t = 0.0
    tard = 0.0
    for j in seq:
        rj, pj, dj = float(inst.r[j]), float(inst.p[j]), float(inst.d[j])
        start = max(t, rj)
        finish = start + pj
        tard += max(0.0, finish - dj)
        t = finish
    return tard

# ---------- 连续位置 -> 排序（random keys） ----------
def keys_to_sequence(keys: np.ndarray) -> np.ndarray:
    # keys: shape [n_jobs]; 返回升序的索引即为加工顺序
    return np.argsort(keys, kind="mergesort")  # 稳定排序，遇到相等不乱序

# ---------- 灰狼优化（GWO）主过程（离散排序版） ----------
def gwo_schedule_min_tardiness(inst: JobInstance,
                               pop_size: int = 40,
                               iters: int = 300,
                               seed: Optional[int] = 42,
                               verbose: bool = True
                               ) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    返回: (最佳顺序, 最小总延期, 最佳keys向量)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = len(inst.r)

    # 初始化狼群：keys ∈ [0,1]^n
    X = np.random.rand(pop_size, n).astype(np.float32)

    # 评估适应度（越小越好）
    def fitness_keys(keys_vec: np.ndarray) -> float:
        seq = keys_to_sequence(keys_vec)
        return sequence_total_tardiness(seq, inst)

    fitness = np.array([fitness_keys(x) for x in X], dtype=np.float32)

    # 选出 alpha、beta、delta（三只最优狼）
    idx_sorted = np.argsort(fitness)
    alpha, beta, delta = X[idx_sorted[0]].copy(), X[idx_sorted[1]].copy(), X[idx_sorted[2]].copy()
    f_alpha, f_beta, f_delta = fitness[idx_sorted[0]], fitness[idx_sorted[1]], fitness[idx_sorted[2]]

    # GWO 主循环
    for t in range(iters):
        # a 从 2 线性减至 0
        a = 2.0 - 2.0 * (t / max(1, iters - 1))

        for i in range(pop_size):
            Xi = X[i]

            # 对 alpha/beta/delta 三只分别计算一次“包围猎物”的更新
            r1, r2 = np.random.rand(n), np.random.rand(n)
            A1 = 2*a*r1 - a
            C1 = 2*r2
            D_alpha = np.abs(C1*alpha - Xi)
            X1 = alpha - A1*D_alpha

            r1, r2 = np.random.rand(n), np.random.rand(n)
            A2 = 2*a*r1 - a
            C2 = 2*r2
            D_beta = np.abs(C2*beta - Xi)
            X2 = beta - A2*D_beta

            r1, r2 = np.random.rand(n), np.random.rand(n)
            A3 = 2*a*r1 - a
            C3 = 2*r2
            D_delta = np.abs(C3*delta - Xi)
            X3 = delta - A3*D_delta

            # 三者平均
            new_Xi = (X1 + X2 + X3) / 3.0

            # 裁剪到 [0,1]
            new_Xi = np.clip(new_Xi, 0.0, 1.0).astype(np.float32)

            # 接受
            X[i] = new_Xi

        # 迭代后统一评估
        fitness = np.array([fitness_keys(x) for x in X], dtype=np.float32)
        idx_sorted = np.argsort(fitness)

        # 更新 alpha/beta/delta
        if fitness[idx_sorted[0]] < f_alpha:
            alpha = X[idx_sorted[0]].copy(); f_alpha = fitness[idx_sorted[0]]
        if fitness[idx_sorted[1]] < f_beta:
            beta  = X[idx_sorted[1]].copy(); f_beta  = fitness[idx_sorted[1]]
        if fitness[idx_sorted[2]] < f_delta:
            delta = X[idx_sorted[2]].copy(); f_delta = fitness[idx_sorted[2]]

        if verbose and (t+1) % max(1, iters//5) == 0:
            print(f"[GWO] iter {t+1}/{iters}  best_tardiness={f_alpha:.3f}")

    best_seq = keys_to_sequence(alpha)
    return best_seq, float(f_alpha), alpha

# ---------- 小演示：与 DQN 贪婪解对比（如果你已经有 agent/greedy_solve） ----------
if __name__ == "__main__":
    # 生成同一个测试实例
    inst: JobInstance = sample_instance(n=10, seed=2)
    print("实例：")
    print("r:", np.round(inst.r, 1))
    print("p:", np.round(inst.p, 1))
    print("d:", np.round(inst.d, 1))

    # 跑 GWO
    best_seq, best_tard, _ = gwo_schedule_min_tardiness(inst, pop_size=50, iters=400, seed=0, verbose=True)
    print("\n[GWO] 最佳顺序:", best_seq.tolist())
    print("[GWO] 最小总延期:", best_tard)
