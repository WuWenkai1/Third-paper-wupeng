# gwo_selection.py
import numpy as np
from typing import Dict, List, Tuple
from parameter import selection_ds  # 需要: ids, release, due, p_tilde, profit, AC, [net_profit]

# ========= 1) 数据适配 =========
def build_data(ds: Dict):
    ids = list(ds["ids"])
    r   = np.array(ds["release"], dtype=float)
    d   = np.array(ds["due"],     dtype=float)
    p   = np.array(ds["p_tilde"], dtype=float)
    if "net_profit" in ds:
        w = np.array(ds["net_profit"], dtype=float)
    else:
        AC = np.array(ds["AC"],     dtype=float)
        prof = np.array(ds["profit"], dtype=float)
        w = prof - AC * p
    return ids, r, d, p, w

# ========= 2) 解码器（SSGS）=========
def decode_schedule_by_perm(perm_idx: List[int], ids: List[int],
                            r: np.ndarray, d: np.ndarray, p: np.ndarray, w: np.ndarray,
                            accept_negative: bool=False):
    """
    perm_idx: 按 '索引(0..n-1)' 的排列
    返回: obj, selected_ids, schedule{ID: (S,C)}
    """
    t = 0.0
    obj = 0.0
    selected = []
    sched = {}
    for idx in perm_idx:
        if (not accept_negative) and (w[idx] <= 0):
            continue
        s = max(t, r[idx])
        c = s + p[idx]
        if c <= d[idx]:
            j = ids[idx]
            selected.append(j)
            sched[j] = (s, c)
            t = c
            obj += w[idx]
    return obj, selected, sched

# ---------- 新增：反射边界，避免堆在 0/1 ----------
#def reflect01(x: np.ndarray) -> np.ndarray:
#    x = np.where(x < 0.0, -x, x)
#    x = np.where(x > 1.0, 2.0 - x, x)
#    return np.clip(x, 0.0, 1.0)

# ---------- 修改：优先级→排列，加入极微 jitter 仅用于平局 ----------
def priority_to_perm(priority: np.ndarray, rng=None) -> List[int]:
    """把优先级向量转换为排列（索引降序，带微抖动打破平局）"""
    if rng is not None:
        # 仅打破相等值的平局；量级极小，不改变总体排序
        pr = priority + 1e-9 * rng.standard_normal(priority.size)
    else:
        pr = priority
    return np.argsort(-pr, kind="mergesort").tolist()

# ---------- GWO 主过程（关键修复：更新后刷新 best_*） ----------
def solve_selection_gwo(ds: Dict,
                        pack_size: int = 40,
                        iterations: int = 400,
                        seed: int = 2025,
                        accept_negative: bool = False,
                        log_every: int = 10):
    rng = np.random.default_rng(seed)
    ids, r, d, p, w = build_data(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0, "best_priority": []}

    # 初始化狼群
    wolves = rng.random((pack_size, n))
    # 启发式1：w/p
    ratio = np.where(p > 0, w / p, -1e9)
    perm1 = np.argsort(-ratio, kind="mergesort")
    wolves[0, perm1] = np.linspace(1.0, 0.0, n)
    # 启发式2：EDD
    perm2 = np.argsort(d, kind="mergesort")
    wolves[1, perm2] = np.linspace(1.0, 0.0, n)

    # 适应度（注意这行改为传 rng，启用平局抖动）
    def fitness_of(prio_vec: np.ndarray) -> float:
        perm = priority_to_perm(prio_vec, rng=rng)
        obj, _, _ = decode_schedule_by_perm(perm, ids, r, d, p, w, accept_negative)
        return obj

    fitness = np.array([fitness_of(wolves[i]) for i in range(pack_size)], dtype=float)

    # 初始全局最好（别依赖迭代内的 alpha 赋值了）
    best_idx = int(np.argmax(fitness))
    best_priority = wolves[best_idx].copy()
    best_obj = float(fitness[best_idx])
    perm = priority_to_perm(best_priority, rng=rng)
    _, best_selected, best_sched = decode_schedule_by_perm(perm, ids, r, d, p, w, accept_negative)

    # GWO 主循环
    a0 = 2.0
    for t_iter in range(1, iterations + 1):
        a = a0 * (1 - t_iter / iterations)

        # 当前 alpha/beta/delta（用于本代更新）
        order = np.argsort(-fitness)
        idx_a, idx_b, idx_c = order[0], order[1], order[2]
        X_a = wolves[idx_a].copy()
        X_b = wolves[idx_b].copy()
        X_c = wolves[idx_c].copy()

        # 更新群体
        for i in range(pack_size):
            X = wolves[i]
            # α 方向
            r1 = rng.random(n); r2 = rng.random(n)
            A1 = 2*a*r1 - a; C1 = 2*r2
            D_a = np.abs(C1 * X_a - X)
            X1  = X_a - A1 * D_a
            # β 方向
            r1 = rng.random(n); r2 = rng.random(n)
            A2 = 2*a*r1 - a; C2 = 2*r2
            D_b = np.abs(C2 * X_b - X)
            X2  = X_b - A2 * D_b
            # δ 方向
            r1 = rng.random(n); r2 = rng.random(n)
            A3 = 2*a*r1 - a; C3 = 2*r2
            D_c = np.abs(C3 * X_c - X)
            X3  = X_c - A3 * D_c

            X_new = (X1 + X2 + X3) / 3.0
            # 微扰 + 反射边界（更温和，避免卡在 0/1）
            sigma = 0.005  # 你也可以用随代次衰减的：0.01*(1 - t_iter/iterations)
            X_new += rng.normal(0.0, sigma, size=n)
            #X_new  = reflect01(X_new)

            # 贪婪替换（允许持平也替换，避免平台卡死）
            f_new = fitness_of(X_new)
            if f_new >= fitness[i]:
                wolves[i] = X_new
                fitness[i] = f_new

        # —— 关键修复：用“更新后的” fitness 刷新全局最好 —— #
        cur_best_idx = int(np.argmax(fitness))
        if fitness[cur_best_idx] > best_obj + 1e-12:
            best_obj = float(fitness[cur_best_idx])
            best_priority = wolves[cur_best_idx].copy()
            perm = priority_to_perm(best_priority, rng=rng)
            _, best_selected, best_sched = decode_schedule_by_perm(perm, ids, r, d, p, w, accept_negative)

        if (log_every is not None) and (t_iter % log_every == 0):
            print(f"[GWO] iter={t_iter:2d}  best_obj={best_obj:.1f}")

    return {
        "selected_ids": best_selected,
        "schedule": best_sched,
        "obj": best_obj,
        "best_priority": best_priority
    }


# ========= 4) 运行示例 =========
if __name__ == "__main__":
    res = solve_selection_gwo(selection_ds,
                              pack_size=50,
                              iterations=4000,
                              seed=10,
                              accept_negative=False,
                              log_every=100)
    print("\n[GWO] 选中的特殊订单ID：", sorted(res["selected_ids"]))
    # 按开始时间输出顺序
    seq = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][0])
    print("[GWO] 加工顺序：")
    for pos, j in enumerate(seq, 1):
        s, c = res["schedule"][j]
        # 辅助读取 r,d 以打印
        idx = selection_ds["ids"].index(j)
        rj = selection_ds["release"][idx]; dj = selection_ds["due"][idx]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | S={s:.1f}, C={c:.1f}")
    print(f"[GWO] 目标(净利润) = {res['obj']:.2f}")
