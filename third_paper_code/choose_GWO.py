# gwo_selection_3machines.py
import numpy as np
from typing import Dict, List, Tuple
from parameter import selection_ds, orders  # 需要: ids, release, due, profit, AC；以及 orders 内含三线工时

# ========= 1) 数据适配（从 orders 提取 P^b, P^e, P^p 并对齐 ids）=========
def build_data_3m(ds: Dict):
    ids = list(ds["ids"])
    r   = np.array(ds["release"], dtype=float)
    d   = np.array(ds["due"],     dtype=float)
    prof= np.array(ds["profit"],  dtype=float)
    AC  = np.array(ds["AC"],      dtype=float)

    # 映射 id -> order 字典
    id2order = {o["id"]: o for o in orders}
    Pb = np.array([float(sum(id2order[j]["proc_body"]))     for j in ids], dtype=float)
    Pe = np.array([float(sum(id2order[j]["proc_cabinet"]))  for j in ids], dtype=float)
    Pp = np.array([float(id2order[j]["proc_pipe"])          for j in ids], dtype=float)

    # 净利润（接单后产生的收益）
    w = prof - AC * (Pb + Pe + Pp)
    return ids, r, d, Pb, Pe, Pp, w

# ========= 2) 解码器（SSGS for 3 machines with SELECTION）=========
def decode_schedule_by_perm_3m(perm_idx: List[int],
                               ids: List[int],
                               r: np.ndarray, d: np.ndarray,
                               Pb: np.ndarray, Pe: np.ndarray, Pp: np.ndarray,
                               w:  np.ndarray,
                               accept_negative: bool=False):
    """
    按给定的“索引排列”perm_idx 进行一次 SSGS：
      - 本体/电柜：各单机，按 perm 顺序并受 r_j 约束排队；
      - 管线：S^p_j = max(C^b_j, C^e_j, t_p)，C^p_j = S^p_j + P^p_j；
      - 若 C^p_j <= d_j 且 (accept_negative 或 w_j>0)：选择该单，并推进 tb/te/tp；
        否则跳过且不占用任何产线时间。
    返回：
      obj（净利润和）, selected_ids, schedule{ID: (Sb,Cb, Se,Ce, Sp,Cp)}
    """
    tb = 0.0  # 本体当前时间
    te = 0.0  # 电柜当前时间
    tp = 0.0  # 装配当前时间

    obj = 0.0
    selected: List[int] = []
    sched: Dict[int, Tuple[float,float,float,float,float,float]] = {}

    for idx in perm_idx:
        j = ids[idx]
        # 先分别计算三线的完成时间（假设接单）
        Sb = max(tb, r[idx]);   Cb = Sb + Pb[idx]
        Se = max(te, r[idx]);   Ce = Se + Pe[idx]
        Sp = max(tp, max(Cb, Ce)); Cp = Sp + Pp[idx]

        feasible = (Cp <= d[idx]) and (accept_negative or w[idx] > 0.0)
        if feasible:
            # 接单：记录 & 推进三条产线时间
            selected.append(j)
            sched[j] = (Sb, Cb, Se, Ce, Sp, Cp)
            tb, te, tp = Cb, Ce, Cp
            obj += w[idx]
        # 若不可行/不盈利：跳过且不消耗任何时间（不修改 tb/te/tp）

    return obj, selected, sched

# ---------- 轻微 jitter 的优先级排序（降序） ----------
def priority_to_perm(priority: np.ndarray, rng=None) -> List[int]:
    if rng is not None:
        pr = priority + 1e-9 * rng.standard_normal(priority.size)
    else:
        pr = priority
    return np.argsort(-pr, kind="mergesort").tolist()

# ========= 3) 灰狼优化（GWO）=========
def solve_selection_gwo_3m(ds: Dict,
                           pack_size: int = 40,
                           iterations: int = 800,
                           seed: int = 2025,
                           accept_negative: bool = False,
                           log_every: int = 50):
    rng = np.random.default_rng(seed)
    ids, r, d, Pb, Pe, Pp, w = build_data_3m(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0, "best_priority": []}

    # --- 初始化狼群（单优先级向量；三线共用）
    wolves = rng.random((pack_size, n))

    # 启发式1：按单位时间收益 w/(Pb+Pe+Pp) 降序
    total_p = Pb + Pe + Pp
    ratio = np.where(total_p > 0, w / total_p, -1e9)
    perm1 = np.argsort(-ratio, kind="mergesort")
    wolves[0, perm1] = np.linspace(1.0, 0.0, n)

    # 启发式2：EDD（交期早者优先）
    perm2 = np.argsort(d, kind="mergesort")
    wolves[1, perm2] = np.linspace(1.0, 0.0, n)

    # 适应度
    def fitness_of(prio_vec: np.ndarray) -> float:
        perm = priority_to_perm(prio_vec, rng=rng)
        obj, _, _ = decode_schedule_by_perm_3m(perm, ids, r, d, Pb, Pe, Pp, w, accept_negative)
        return obj

    fitness = np.array([fitness_of(wolves[i]) for i in range(pack_size)], dtype=float)

    # 全局最优
    best_idx = int(np.argmax(fitness))
    best_priority = wolves[best_idx].copy()
    best_obj = float(fitness[best_idx])
    perm = priority_to_perm(best_priority, rng=rng)
    _, best_selected, best_sched = decode_schedule_by_perm_3m(perm, ids, r, d, Pb, Pe, Pp, w, accept_negative)

    a0 = 2.0
    for t_iter in range(1, iterations + 1):
        a = a0 * (1 - t_iter / iterations)  # GWO 控制参数

        order = np.argsort(-fitness)
        idx_a, idx_b, idx_c = order[0], order[1], order[2]
        X_a = wolves[idx_a].copy()
        X_b = wolves[idx_b].copy()
        X_c = wolves[idx_c].copy()

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
            # 微扰 + 截断
            sigma = 0.01 * (1 - t_iter / iterations)  # 逐渐减小
            X_new += rng.normal(0.0, sigma, size=n)
            X_new = np.clip(X_new, 0.0, 1.0)

            f_new = fitness_of(X_new)
            if f_new >= fitness[i]:
                wolves[i] = X_new
                fitness[i] = f_new

        # 刷新全局最优
        cur_best = int(np.argmax(fitness))
        if fitness[cur_best] > best_obj + 1e-12:
            best_obj = float(fitness[cur_best])
            best_priority = wolves[cur_best].copy()
            perm = priority_to_perm(best_priority, rng=rng)
            _, best_selected, best_sched = decode_schedule_by_perm_3m(
                perm, ids, r, d, Pb, Pe, Pp, w, accept_negative
            )

        if (log_every is not None) and (t_iter % log_every == 0):
            print(f"[GWO-3M] iter={t_iter:4d}  best_obj={best_obj:.1f}")

    return {
        "selected_ids": best_selected,
        "schedule": best_sched,        # {ID: (Sb,Cb, Se,Ce, Sp,Cp)}
        "obj": best_obj,
        "best_priority": best_priority
    }

# ========= 4) 运行示例 =========
if __name__ == "__main__":
    res = solve_selection_gwo_3m(selection_ds,
                                 pack_size=60,
                                 iterations=2000,
                                 seed=10,
                                 accept_negative=False,
                                 log_every=100)
    print("\n[GWO-3M] 选中的特殊订单ID：", sorted(res["selected_ids"]))
    # 按装配开始时间输出顺序
    seq = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][4])  # Sp 在 tuple 第 5 位
    print("[GWO-3M] 加工顺序（按装配开始）:")
    for pos, j in enumerate(seq, 1):
        Sb,Cb, Se,Ce, Sp,Cp = res["schedule"][j]
        idx = selection_ds["ids"].index(j)
        rj = selection_ds["release"][idx]; dj = selection_ds["due"][idx]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | "
              f"Sb={Sb:.1f}, Cb={Cb:.1f} ; Se={Se:.1f}, Ce={Ce:.1f} ; Sp={Sp:.1f}, Cp={Cp:.1f}")
    print(f"[GWO-3M] 目标(净利润) = {res['obj']:.2f}")
