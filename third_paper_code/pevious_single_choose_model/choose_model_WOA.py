# woa_selection.py
import numpy as np
from typing import Dict, List, Optional
from parameter import selection_ds  # 需要: ids, release, due, p_tilde, profit, AC, [net_profit]

# ========= 数据适配 =========
def build_data(ds: Dict):
    ids = list(ds["ids"])
    r   = np.array(ds["release"], dtype=float)
    d   = np.array(ds["due"],     dtype=float)
    p   = np.array(ds["p_tilde"], dtype=float)
    if "net_profit" in ds:
        w = np.array(ds["net_profit"], dtype=float)
    else:
        AC   = np.array(ds["AC"],     dtype=float)
        prof = np.array(ds["profit"], dtype=float)
        w = prof - AC * p
    return ids, r, d, p, w

# ========= 解码器（SSGS）=========
def decode_schedule_by_perm(perm_idx: List[int], ids: List[int],
                            r: np.ndarray, d: np.ndarray, p: np.ndarray, w: np.ndarray,
                            accept_negative: bool=False):
    """
    串行放置：s=max(t,r), c=s+p；若 c<=d 且 w>0(可配置) 则接纳
    返回: obj, selected_ids, schedule{ID:(S,C)}
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

def priority_to_perm(priority: np.ndarray) -> List[int]:
    """优先级 -> 排列（降序，稳定排序）"""
    return np.argsort(-priority, kind="mergesort").tolist()

# ========= WOA 主过程 =========
def solve_selection_woa(ds: Dict,
                        pop_size: int = 40,
                        iterations: int = 400,
                        b_spiral: float = 1.0,      # 螺旋常数 b
                        seed: int = 2025,
                        accept_negative: bool = False,
                        log_every: Optional[int] = 50):
    """
    连续版 WOA：群体个体为 [0,1]^n 的优先级向量
    更新规则：
      a 从 2 线性降到 0；A = 2a*r - a；C = 2*r；p~U(0,1)
      p<0.5 且 |A|<1：围猎（exploit）；p<0.5 且 |A|>=1：探索（explore）
      p>=0.5：螺旋更新
    """
    rng = np.random.default_rng(seed)
    ids, r, d, p, w = build_data(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0, "best_priority": []}

    lb, ub = 0.0, 1.0

    # 初始化群体：随机 + 两个启发式（w/p，EDD）
    whales = rng.random((pop_size, n))
    ratio = np.where(p > 0, w / p, -1e9)
    perm1 = np.argsort(-ratio, kind="mergesort")        # w/p 降序
    whales[0, perm1] = np.linspace(1.0, 0.0, n)
    perm2 = np.argsort(d, kind="mergesort")             # EDD
    whales[1, perm2] = np.linspace(1.0, 0.0, n)

    def fitness_of(prio_vec: np.ndarray) -> float:
        perm = priority_to_perm(prio_vec)
        obj, _, _ = decode_schedule_by_perm(perm, ids, r, d, p, w, accept_negative)
        return obj

    fitness = np.array([fitness_of(whales[i]) for i in range(pop_size)], dtype=float)
    best_idx = int(np.argmax(fitness))
    best = whales[best_idx].copy()
    best_obj = fitness[best_idx]

    for t in range(1, iterations + 1):
        a = 2.0 - 2.0 * t / iterations   # a: 2 -> 0 线性递减

        for i in range(pop_size):
            X = whales[i]
            r1 = rng.random(n)
            r2 = rng.random(n)
            A = 2*a*r1 - a
            Cc = 2*r2
            p_rand = rng.random()

            if p_rand < 0.5:
                if np.all(np.abs(A) < 1.0):
                    # 围猎（利用）：X' = best - A*|C*best - X|
                    D = np.abs(Cc * best - X)
                    X_new = best - A * D
                else:
                    # 探索：X' = X_rand - A*|C*X_rand - X|
                    X_rand = whales[rng.integers(0, pop_size)]
                    D = np.abs(Cc * X_rand - X)
                    X_new = X_rand - A * D
            else:
                # 螺旋：X' = D' * exp(b*l) * cos(2πl) + best
                Dp = np.abs(best - X)
                l = rng.uniform(-1.0, 1.0, size=n)
                X_new = Dp * np.exp(b_spiral * l) * np.cos(2*np.pi*l) + best

            # 轻微噪声，帮助跳出平台
            X_new += rng.normal(0.0, 0.01, size=n)
            # 截断
            X_new = np.clip(X_new, lb, ub)

            # 选择性替换（贪婪）
            f_new = fitness_of(X_new)
            if f_new > fitness[i] + 1e-12:
                whales[i] = X_new
                fitness[i] = f_new
                if f_new > best_obj + 1e-12:
                    best_obj = f_new
                    best = X_new.copy()

        if (log_every is not None) and (t % log_every == 0):
            print(f"[WOA] iter={t:4d}  best_obj={best_obj:.2f}")

    # 最终解码
    perm = priority_to_perm(best)
    obj, selected, sched = decode_schedule_by_perm(perm, ids, r, d, p, w, accept_negative)
    return {
        "selected_ids": selected,
        "schedule": sched,           # {ID: (S,C)}
        "obj": obj,
        "best_priority": best.tolist()
    }

# ========= 运行示例 =========
if __name__ == "__main__":
    res = solve_selection_woa(selection_ds,
                              pop_size=40,
                              iterations=4000,
                              b_spiral=1.0,
                              seed=2025,
                              accept_negative=False,
                              log_every=50)
    print("\n[WOA] 选中的特殊订单ID：", sorted(res["selected_ids"]))
    seq = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][0])
    print("[WOA] 加工顺序：")
    for pos, j in enumerate(seq, 1):
        s, c = res["schedule"][j]
        idx = selection_ds["ids"].index(j)
        rj, dj = selection_ds["release"][idx], selection_ds["due"][idx]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | S={s:.1f}, C={c:.1f}")
    print(f"[WOA] 目标(净利润) = {res['obj']:.2f}")
