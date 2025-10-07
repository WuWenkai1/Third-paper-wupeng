# ga_selection.py
import numpy as np
from typing import Dict, List, Tuple
from parameter import selection_ds  # 需要你已有的 {ids, release, due, p_tilde, profit, AC, [net_profit]}

# ========== 1) 数据适配 ==========
def build_data(ds: Dict):
    ids = list(ds["ids"])
    r   = {j: float(v) for j, v in zip(ids, ds["release"])}
    d   = {j: float(v) for j, v in zip(ids, ds["due"])}
    p   = {j: float(v) for j, v in zip(ids, ds["p_tilde"])}
    if "net_profit" in ds:
        w = {j: float(v) for j, v in zip(ids, ds["net_profit"])}
    else:
        AC = {j: float(v) for j, v in zip(ids, ds["AC"])}
        prof = {j: float(v) for j, v in zip(ids, ds["profit"])}
        w = {j: prof[j] - AC[j]*p[j] for j in ids}
    return ids, r, d, p, w

# ========== 2) 解码（从排列得到选择与排程） ==========
def decode_schedule(permutation: List[int], r: Dict, d: Dict, p: Dict, w: Dict,
                    accept_negative: bool=False):
    """
    给定一个订单排列，按顺序尝试调度：
      s_j = max(current_time, r_j)
      若 s_j + p_j <= d_j 且 (w_j>0 或 accept_negative=True) 则选择并排入，否则跳过。
    返回: obj, selected_ids, schedule{j:(S,C)}
    """
    t = 0.0
    obj = 0.0
    selected = []
    sched = {}
    for j in permutation:
        if (not accept_negative) and (w[j] <= 0):
            continue
        s = max(t, r[j])
        c = s + p[j]
        if c <= d[j]:
            selected.append(j)
            sched[j] = (s, c)
            t = c
            obj += w[j]
        # else: 跳过（不可行或超期）
    return obj, selected, sched

# ========== 3) 遗传组件 ==========
def tournament_select(pop, fitness, rng, k=3):
    idxs = rng.integers(0, len(pop), size=k)
    best = max(idxs, key=lambda i: fitness[i])
    return pop[best].copy()

def ox_crossover(parent1, parent2, rng):
    """Order Crossover (OX) 适用于排列编码"""
    n = len(parent1)
    a, b = sorted(rng.integers(0, n, size=2))
    child = [-1]*n
    # 保留片段
    child[a:b+1] = parent1[a:b+1]
    # 填充剩余
    ptr = (b+1) % n
    for x in parent2:
        if x not in child:
            child[ptr] = x
            ptr = (ptr+1) % n
    return child

def swap_mutation(perm, rng, pm=0.2):
    if rng.random() < pm:
        n = len(perm)
        i, j = rng.integers(0, n, size=2)
        perm[i], perm[j] = perm[j], perm[i]
    return perm

# ========== 4) 主过程 ==========
def solve_selection_ga(ds: Dict,
                       pop_size=80,
                       generations=400,
                       elite_ratio=0.1,
                       tournament_k=3,
                       crossover_rate=0.9,
                       mutation_rate=0.2,
                       accept_negative=False,
                       seed=42,
                       log_every=50):
    rng = np.random.default_rng(seed)
    ids, r, d, p, w = build_data(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0, "best_perm": []}

    # 初始种群：随机排列
    population = [rng.permutation(ids).tolist() for _ in range(pop_size)]

    # 适应度
    def eval_perm(perm):
        obj, _, _ = decode_schedule(perm, r, d, p, w, accept_negative=accept_negative)
        return obj

    fitness = [eval_perm(ind) for ind in population]
    elite_num = max(1, int(pop_size * elite_ratio))

    best_obj = -1e30
    best_perm = None
    best_selected, best_sched = [], {}

    for gen in range(1, generations+1):
        # 精英保留
        elite_idx = np.argsort(fitness)[-elite_num:][::-1]
        elites = [population[i].copy() for i in elite_idx]
        new_pop = elites.copy()

        # 生成后代
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitness, rng, k=tournament_k)
            p2 = tournament_select(population, fitness, rng, k=tournament_k)
            child = p1
            if rng.random() < crossover_rate:
                child = ox_crossover(p1, p2, rng)
            child = swap_mutation(child, rng, pm=mutation_rate)
            new_pop.append(child)

        population = new_pop
        fitness = [eval_perm(ind) for ind in population]

        # 更新最好解
        idx = int(np.argmax(fitness))
        if fitness[idx] > best_obj + 1e-9:
            best_obj = fitness[idx]
            best_perm = population[idx].copy()
            _, best_selected, best_sched = decode_schedule(best_perm, r, d, p, w,
                                                           accept_negative=accept_negative)
        if (log_every is not None) and (gen % log_every == 0):
            print(f"[GA] gen={gen:4d}  best_obj={best_obj:.2f}")

    return {
        "selected_ids": best_selected,
        "schedule": best_sched,   # {j:(S,C)}
        "obj": best_obj,
        "best_perm": best_perm
    }

# ========== 5) 运行示例 ==========
if __name__ == "__main__":
    res = solve_selection_ga(selection_ds,
                             pop_size=80,
                             generations=400,
                             elite_ratio=0.1,
                             tournament_k=3,
                             crossover_rate=0.9,
                             mutation_rate=0.2,
                             accept_negative=False,  # 只选净利润>0的单
                             seed=2025,
                             log_every=50)
    print("\n[GA] 选中的特殊订单ID：", sorted(res["selected_ids"]))
    # 按开始时间输出顺序
    sequence = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][0])
    print("[GA] 加工顺序：")
    for pos, j in enumerate(sequence, 1):
        s, c = res["schedule"][j]
        rj, dj = selection_ds["release"][selection_ds["ids"].index(j)], selection_ds["due"][selection_ds["ids"].index(j)]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | S={s:.1f}, C={c:.1f}")
    print(f"[GA] 目标(净利润) = {res['obj']:.2f}")
