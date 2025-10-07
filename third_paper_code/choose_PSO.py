# pso_selection_3machines.py
import numpy as np
from typing import Dict, List
from parameter import selection_ds, orders

# ========= 数据提取 =========
def build_data_3m(ds: Dict):
    ids = list(ds["ids"])
    r   = np.array(ds["release"], dtype=float)
    d   = np.array(ds["due"], dtype=float)
    prof= np.array(ds["profit"],  dtype=float)
    AC  = np.array(ds["AC"],      dtype=float)

    id2order = {o["id"]: o for o in orders}
    Pb = np.array([float(sum(id2order[j]["proc_body"]))    for j in ids])
    Pe = np.array([float(sum(id2order[j]["proc_cabinet"])) for j in ids])
    Pp = np.array([float(id2order[j]["proc_pipe"])         for j in ids])
    w  = prof - AC * (Pb + Pe + Pp)
    return ids, r, d, Pb, Pe, Pp, w

# ========= 三机解码 =========
def decode_schedule_by_perm_3m(perm_ids: list[int],
                               ids: list[int],
                               r: np.ndarray, d: np.ndarray,
                               Pb: np.ndarray, Pe: np.ndarray, Pp: np.ndarray,
                               w: np.ndarray,
                               accept_negative: bool=False):
    id2idx = {j:i for i,j in enumerate(ids)}
    tb = te = tp = 0.0
    obj = 0.0
    selected = []
    sched = {}
    for j in perm_ids:
        idx = id2idx[j]
        Sb = max(tb, r[idx]); Cb = Sb + Pb[idx]
        Se = max(te, r[idx]); Ce = Se + Pe[idx]
        Sp = max(tp, max(Cb, Ce)); Cp = Sp + Pp[idx]
        feasible = (Cp <= d[idx]) and (accept_negative or w[idx] > 0.0)
        if feasible:
            selected.append(j)
            sched[j] = (Sb,Cb, Se,Ce, Sp,Cp)
            tb, te, tp = Cb, Ce, Cp
            obj += w[idx]
    return obj, selected, sched

# ========= 工具函数 =========
def priority_to_perm(priority: np.ndarray) -> list[int]:
    """优先级→排列"""
    return np.argsort(-priority, kind="mergesort").tolist()

# ========= 粒子群算法 PSO =========
def solve_selection_pso_3m(ds: Dict,
                           swarm_size=60,
                           iterations=800,
                           w_inertia=0.7,
                           c1=1.5, c2=1.5,
                           seed=2025,
                           accept_negative=False,
                           log_every=50):
    rng = np.random.default_rng(seed)
    ids, r, d, Pb, Pe, Pp, w = build_data_3m(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0}

    # 初始化粒子群
    X = rng.random((swarm_size, n))
    V = np.zeros_like(X)
    P_best = X.copy()
    fitness = np.zeros(swarm_size)

    def fitness_of(x):
        perm = priority_to_perm(x)
        order = [ids[k] for k in perm]
        obj, _, _ = decode_schedule_by_perm_3m(order, ids, r, d, Pb, Pe, Pp, w, accept_negative)
        return obj

    # 初始化适应度
    for i in range(swarm_size):
        fitness[i] = fitness_of(X[i])
    G_best = X[np.argmax(fitness)].copy()
    best_obj = np.max(fitness)

    # 迭代更新
    for t in range(1, iterations+1):
        for i in range(swarm_size):
            r1, r2 = rng.random(n), rng.random(n)
            V[i] = w_inertia * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])
            X[i] = np.clip(X[i] + V[i], 0.0, 1.0)

            f = fitness_of(X[i])
            if f > fitness[i]:
                fitness[i] = f
                P_best[i] = X[i].copy()
        # 更新全局最优
        idx = np.argmax(fitness)
        if fitness[idx] > best_obj + 1e-9:
            best_obj = fitness[idx]
            G_best = X[idx].copy()

        if log_every and t % log_every == 0:
            print(f"[PSO-3M] iter={t:4d}  best_obj={best_obj:.2f}")

    # 解码最优解
    perm = priority_to_perm(G_best)
    order = [ids[k] for k in perm]
    _, selected, sched = decode_schedule_by_perm_3m(order, ids, r, d, Pb, Pe, Pp, w, accept_negative)
    return {
        "selected_ids": selected,
        "schedule": sched,
        "obj": best_obj
    }

# ========= 主程序示例 =========
if __name__ == "__main__":
    res = solve_selection_pso_3m(selection_ds,
                                 swarm_size=80,
                                 iterations=1000,
                                 w_inertia=0.7,
                                 c1=1.6, c2=1.6,
                                 seed=2025,
                                 accept_negative=False,
                                 log_every=50)
    print("\n[PSO-3M] 选中的特殊订单ID：", sorted(res["selected_ids"]))
    seq = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][4])  # 按装配开始
    print("[PSO-3M] 加工顺序：")
    for pos, j in enumerate(seq, 1):
        Sb,Cb, Se,Ce, Sp,Cp = res["schedule"][j]
        idx = selection_ds["ids"].index(j)
        rj, dj = selection_ds["release"][idx], selection_ds["due"][idx]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | "
              f"Sb={Sb:.1f},Cb={Cb:.1f}; Se={Se:.1f},Ce={Ce:.1f}; Sp={Sp:.1f},Cp={Cp:.1f}")
    print(f"[PSO-3M] 目标(净利润) = {res['obj']:.2f}")
