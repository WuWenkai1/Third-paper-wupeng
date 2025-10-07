# woa_selection_3machines.py
import numpy as np
from typing import Dict, List, Tuple
from parameter import selection_ds, orders  # 需要：selection_ds(含 ids, release, due, profit, AC)，以及 orders(含三线工时)

# =========================
# 1) 数据适配（取三条线工时 & 计算净利润）
# =========================
def build_data_3m(ds: Dict):
    ids = list(ds["ids"])
    r   = np.array(ds["release"], dtype=float)
    d   = np.array(ds["due"],     dtype=float)
    prof= np.array(ds["profit"],  dtype=float)
    AC  = np.array(ds["AC"],      dtype=float)

    id2order = {o["id"]: o for o in orders}
    Pb = np.array([float(sum(id2order[j]["proc_body"]))     for j in ids], dtype=float)
    Pe = np.array([float(sum(id2order[j]["proc_cabinet"]))  for j in ids], dtype=float)
    Pp = np.array([float(id2order[j]["proc_pipe"])          for j in ids], dtype=float)

    # 净利润（接单后产生的收益）
    w = prof - AC * (Pb + Pe + Pp)
    return ids, r, d, Pb, Pe, Pp, w

# =========================
# 2) 三机 SSGS 解码（带“选择或跳过”逻辑）
# =========================
def decode_schedule_by_perm_3m(perm_idx: List[int],
                               ids: List[int],
                               r: np.ndarray, d: np.ndarray,
                               Pb: np.ndarray, Pe: np.ndarray, Pp: np.ndarray,
                               w:  np.ndarray,
                               accept_negative: bool=False):
    """
    根据 perm_idx 的顺序尝试接单：
      - 本体/电柜各单机：S_b=max(t_b,r), C_b=S_b+P_b；S_e=max(t_e,r), C_e=S_e+P_e
      - 装配线：S_p=max(t_p, C_b, C_e), C_p=S_p+P_p
      - 若 C_p<=d 且(accept_negative或w>0)，则接单并推进 t_b,t_e,t_p；否则跳过且不耗时
    返回：
      obj（净利润）, selected_ids, schedule{ID: (Sb,Cb, Se,Ce, Sp,Cp)}
    """
    tb = te = tp = 0.0
    obj = 0.0
    selected: List[int] = []
    sched: Dict[int, Tuple[float,float,float,float,float,float]] = {}

    for idx in perm_idx:
        j = ids[idx]
        Sb = max(tb, r[idx]);   Cb = Sb + Pb[idx]
        Se = max(te, r[idx]);   Ce = Se + Pe[idx]
        Sp = max(tp, max(Cb, Ce)); Cp = Sp + Pp[idx]

        feasible = (Cp <= d[idx]) and (accept_negative or w[idx] > 0.0)
        if feasible:
            selected.append(j)
            sched[j] = (Sb, Cb, Se, Ce, Sp, Cp)
            tb, te, tp = Cb, Ce, Cp
            obj += w[idx]
        # 不可行/不盈利：跳过（不推进时间）

    return obj, selected, sched

# =========================
# 3) 优先级→排列（降序；用极微抖动打破平局）
# =========================
def priority_to_perm(priority: np.ndarray, rng=None) -> List[int]:
    if rng is not None:
        pr = priority + 1e-9 * rng.standard_normal(priority.size)
    else:
        pr = priority
    return np.argsort(-pr, kind="mergesort").tolist()

# =========================
# 4) 鲸鱼算法（WOA）主过程
# =========================
def solve_selection_woa_3m(ds: Dict,
                           pack_size: int = 50,
                           iterations: int = 1500,
                           seed: int = 2025,
                           accept_negative: bool = False,
                           log_every: int = 50):
    """
    连续编码：长度 n 的优先级向量 x∈[0,1]^n
    更新规则：WOA 三种机制（围捕、螺旋、搜索）
    """
    rng = np.random.default_rng(seed)
    ids, r, d, Pb, Pe, Pp, w = build_data_3m(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0, "best_priority": []}

    # --- 初始化种群（含两只启发式个体）
    pop = rng.random((pack_size, n))
    fit = np.empty(pack_size, dtype=float)

    # 启发式1：单位时间收益 ratio = w / (Pb+Pe+Pp)
    tot_p = Pb + Pe + Pp
    ratio = np.where(tot_p > 0, w / tot_p, -1e9)
    perm1 = np.argsort(-ratio, kind="mergesort")
    pop[0, perm1] = np.linspace(1.0, 0.0, n)

    # 启发式2：EDD
    perm2 = np.argsort(d, kind="mergesort")
    pop[1, perm2] = np.linspace(1.0, 0.0, n)

    def fitness_of(prio_vec: np.ndarray) -> float:
        perm = priority_to_perm(prio_vec, rng=rng)
        obj, _, _ = decode_schedule_by_perm_3m(perm, ids, r, d, Pb, Pe, Pp, w, accept_negative)
        return obj

    for i in range(pack_size):
        fit[i] = fitness_of(pop[i])

    # 全局最优
    best_idx = int(np.argmax(fit))
    X_best = pop[best_idx].copy()
    f_best = float(fit[best_idx])
    # 保存一份 best 的可读结果
    perm = priority_to_perm(X_best, rng=rng)
    _, best_selected, best_sched = decode_schedule_by_perm_3m(perm, ids, r, d, Pb, Pe, Pp, w, accept_negative)

    # --- 主循环
    b = 1.0  # WOA 螺旋常数
    for t in range(1, iterations + 1):
        # a 从 2 → 0 线性下降
        a = 2.0 * (1 - t / iterations)

        for i in range(pack_size):
            X = pop[i]
            r1 = rng.random(n)
            r2 = rng.random(n)
            A  = 2 * a * r1 - a
            C  = 2 * r2
            p  = rng.random()          # 选择概率（标量）
            l  = rng.uniform(-1.0, 1.0)  # 螺旋参数（标量）

            if p < 0.5:
                if np.all(np.abs(A) < 1.0):
                    # Exploitation：围绕当前最优
                    D = np.abs(C * X_best - X)
                    X_new = X_best - A * D
                else:
                    # Exploration：参考随机个体
                    rand_idx = rng.integers(0, pack_size)
                    X_rand = pop[rand_idx]
                    D = np.abs(C * X_rand - X)
                    X_new = X_rand - A * D
            else:
                # Bubble-net：螺旋更新
                D_prime = np.abs(X_best - X)
                X_new = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + X_best

            # 轻微高斯扰动 + 截断
            sigma = 0.01 * (1 - t / iterations)  # 随代次收敛
            X_new += rng.normal(0.0, sigma, size=n)
            X_new = np.clip(X_new, 0.0, 1.0)

            f_new = fitness_of(X_new)
            # 贪婪替换（允许持平替换，便于爬出平台）
            if f_new >= fit[i]:
                pop[i] = X_new
                fit[i] = f_new

        # 刷新全局最优
        cur_best = int(np.argmax(fit))
        if fit[cur_best] > f_best + 1e-12:
            f_best = float(fit[cur_best])
            X_best = pop[cur_best].copy()
            perm = priority_to_perm(X_best, rng=rng)
            _, best_selected, best_sched = decode_schedule_by_perm_3m(
                perm, ids, r, d, Pb, Pe, Pp, w, accept_negative
            )

        if (log_every is not None) and (t % log_every == 0):
            print(f"[WOA-3M] iter={t:4d}  best_obj={f_best:.1f}")

    return {
        "selected_ids": best_selected,
        "schedule": best_sched,   # {ID: (Sb,Cb, Se,Ce, Sp,Cp)}
        "obj": f_best,
        "best_priority": X_best
    }

# =========================
# 5) 运行示例
# =========================
if __name__ == "__main__":
    res = solve_selection_woa_3m(selection_ds,
                                 pack_size=60,
                                 iterations=2000,
                                 seed=10,
                                 accept_negative=False,
                                 log_every=100)
    print("\n[WOA-3M] 选中的特殊订单ID：", sorted(res["selected_ids"]))

    # 按装配开始时间输出顺序
    seq = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][4])  # Sp 是第5个
    print("[WOA-3M] 加工顺序（按装配开始）:")
    for pos, j in enumerate(seq, 1):
        Sb,Cb, Se,Ce, Sp,Cp = res["schedule"][j]
        # 若想把编号映射为 1..(特殊订单数)，可减去普通订单数
        # true_id = j - num_normal_orders
        idx = selection_ds["ids"].index(j)
        rj = selection_ds["release"][idx]; dj = selection_ds["due"][idx]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | "
              f"Sb={Sb:.1f}, Cb={Cb:.1f} ; Se={Se:.1f}, Ce={Ce:.1f} ; Sp={Sp:.1f}, Cp={Cp:.1f}")
    print(f"[WOA-3M] 目标(净利润) = {res['obj']:.2f}")
