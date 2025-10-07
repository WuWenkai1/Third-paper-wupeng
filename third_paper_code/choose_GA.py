# ga_selection_3machines.py
import numpy as np
from typing import Dict, List, Tuple
from parameter import selection_ds, orders  # 需要: selection_ds(ids, release, due, profit, AC), orders(含三线工时)

# ========== 1) 数据适配：从 orders 提三条线工时，算净利润 ==========
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

    w  = prof - AC * (Pb + Pe + Pp)  # 净利润（被选且按时完成后才计入）
    return ids, r, d, Pb, Pe, Pp, w

# ========== 2) 三机 SSGS 解码（选择或跳过）==========
def decode_schedule_by_perm_3m(perm_ids: List[int],
                               ids: List[int],
                               r: np.ndarray, d: np.ndarray,
                               Pb: np.ndarray, Pe: np.ndarray, Pp: np.ndarray,
                               w:  np.ndarray,
                               accept_negative: bool=False):
    """
    perm_ids: 直接是订单ID的排列（不是索引）。内部会映射到数组索引。
    规则：
      - Body/Cabinet 各单机并行：S_b=max(t_b, r), C_b=S_b+P_b；S_e=max(t_e, r), C_e=S_e+P_e
      - Pipeline 汇合：S_p=max(t_p, C_b, C_e), C_p=S_p+P_p
      - 若 C_p<=d 且(accept_negative 或 w>0) 则接单并推进三线时间；否则跳过且不占时
    返回: obj, selected_ids, schedule{ID: (Sb,Cb, Se,Ce, Sp,Cp)}
    """
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
            sched[j] = (Sb, Cb, Se, Ce, Sp, Cp)
            tb, te, tp = Cb, Ce, Cp
            obj += w[idx]

    return obj, selected, sched

# ========== 3) GA 组件：选择 / 交叉 / 变异 / 启发式初始种群 ==========
def tournament_select(pop, fitness, rng, k=3):
    idxs = rng.integers(0, len(pop), size=k)
    best = max(idxs, key=lambda i: fitness[i])
    return pop[best].copy()

def ox_crossover(parent1: List[int], parent2: List[int], rng) -> List[int]:
    """Order Crossover (OX)，保持排列合法"""
    n = len(parent1)
    a, b = sorted(rng.integers(0, n, size=2))
    child = [-1]*n
    child[a:b+1] = parent1[a:b+1]
    ptr = (b+1) % n
    for x in parent2:
        if x not in child:
            child[ptr] = x
            ptr = (ptr+1) % n
    return child

def swap_mutation(perm: List[int], rng, pm=0.2) -> List[int]:
    if rng.random() < pm:
        n = len(perm); i, j = rng.integers(0, n, size=2)
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def insert_mutation(perm: List[int], rng, pm=0.2) -> List[int]:
    if rng.random() < pm:
        n = len(perm); i, j = rng.integers(0, n, size=2)
        if i == j: j = (j+1) % n
        a, b = (i, j) if i < j else (j, i)
        val = perm[a]
        del perm[a]
        perm.insert(b, val)
    return perm

def build_seed_population(ids, r, d, Pb, Pe, Pp, w, pop_size, rng):
    """启发式种子：1) 单位时间利润降序；2) EDD；3) 随机若干"""
    population = []
    # 1) 单位时间利润 w/(Pb+Pe+Pp) 降序
    totP = Pb + Pe + Pp
    ratio = np.where(totP>0, w/totP, -1e9)
    ids_by_ratio = [x for _,x in sorted(zip(-ratio, ids))]
    population.append(ids_by_ratio)
    # 2) EDD（最早交期优先）
    ids_by_edd = [x for _,x in sorted(zip(d, ids))]
    population.append(ids_by_edd)
    # 3) 随机若干
    while len(population) < pop_size:
        population.append(rng.permutation(ids).tolist())
    return population

# ========== 4) 主过程：GA for 3-machine selection ==========
def solve_selection_ga_3m(ds: Dict,
                          pop_size=100,
                          generations=600,
                          elite_ratio=0.1,
                          tournament_k=3,
                          crossover_rate=0.9,
                          mutation_rate_swap=0.2,
                          mutation_rate_insert=0.2,
                          immigrants_ratio=0.05,   # 随机移民比例，保持多样性
                          accept_negative=False,
                          seed=42,
                          log_every=50):
    rng = np.random.default_rng(seed)
    ids, r, d, Pb, Pe, Pp, w = build_data_3m(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0, "best_perm": []}

    # 初始种群（含启发式种子）
    population = build_seed_population(ids, r, d, Pb, Pe, Pp, w, pop_size, rng)

    def eval_perm(perm_ids: List[int]):
        obj, _, _ = decode_schedule_by_perm_3m(perm_ids, ids, r, d, Pb, Pe, Pp, w, accept_negative)
        return obj

    fitness = [eval_perm(ind) for ind in population]
    elite_num = max(1, int(pop_size * elite_ratio))

    best_obj = -1e30
    best_perm = None
    best_selected, best_sched = [], {}

    for gen in range(1, generations+1):
        # —— 精英保留 ——
        elite_idx = np.argsort(fitness)[-elite_num:][::-1]
        elites = [population[i].copy() for i in elite_idx]
        new_pop = elites.copy()

        # —— 产生子代 ——
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitness, rng, k=tournament_k)
            p2 = tournament_select(population, fitness, rng, k=tournament_k)
            child = p1
            if rng.random() < crossover_rate:
                child = ox_crossover(p1, p2, rng)
            # 双变异：swap + insert，提升多样性
            child = swap_mutation(child, rng, pm=mutation_rate_swap)
            child = insert_mutation(child, rng, pm=mutation_rate_insert)
            new_pop.append(child)

        # —— 随机移民（防早熟）——
        num_imm = int(pop_size * immigrants_ratio)
        for _ in range(num_imm):
            new_pop[rng.integers(elite_num, pop_size)] = rng.permutation(ids).tolist()

        population = new_pop
        fitness = [eval_perm(ind) for ind in population]

        # —— 更新最好解 ——
        idx = int(np.argmax(fitness))
        if fitness[idx] > best_obj + 1e-9:
            best_obj = fitness[idx]
            best_perm = population[idx].copy()
            _, best_selected, best_sched = decode_schedule_by_perm_3m(best_perm, ids, r, d, Pb, Pe, Pp, w,
                                                                      accept_negative=accept_negative)
        if (log_every is not None) and (gen % log_every == 0):
            print(f"[GA-3M] gen={gen:4d}  best_obj={best_obj:.2f}")

    return {
        "selected_ids": best_selected,
        "schedule": best_sched,   # {ID: (Sb,Cb, Se,Ce, Sp,Cp)}
        "obj": best_obj,
        "best_perm": best_perm
    }

# ========== 5) 运行示例 ==========
if __name__ == "__main__":
    res = solve_selection_ga_3m(selection_ds,
                                pop_size=120,
                                generations=800,
                                elite_ratio=0.12,
                                tournament_k=4,
                                crossover_rate=0.92,
                                mutation_rate_swap=0.25,
                                mutation_rate_insert=0.25,
                                immigrants_ratio=0.06,
                                accept_negative=False,
                                seed=2025,
                                log_every=50)
    print("\n[GA-3M] 选中的特殊订单ID：", sorted(res["selected_ids"]))
    # 按装配开始时间输出顺序（Sp 是 tuple 第 5 位）
    seq = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][4])
    print("[GA-3M] 加工顺序：")
    for pos, j in enumerate(seq, 1):
        Sb,Cb, Se,Ce, Sp,Cp = res["schedule"][j]
        idx = selection_ds["ids"].index(j)
        rj, dj = selection_ds["release"][idx], selection_ds["due"][idx]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | "
              f"Sb={Sb:.1f}, Cb={Cb:.1f} ; Se={Se:.1f}, Ce={Ce:.1f} ; Sp={Sp:.1f}, Cp={Cp:.1f}")
    print(f"[GA-3M] 目标(净利润) = {res['obj']:.2f}")
