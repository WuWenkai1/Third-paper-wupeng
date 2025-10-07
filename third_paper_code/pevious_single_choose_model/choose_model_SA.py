# sa_selection_3machines.py
import numpy as np
from typing import Dict, List, Tuple
from parameter import selection_ds, orders  # 使用三机订单结构

# ========== 数据读取 ==========
def build_data_3m(ds: Dict):
    ids = list(ds["ids"])
    r   = np.array(ds["release"], dtype=float)
    d   = np.array(ds["due"],     dtype=float)
    prof= np.array(ds["profit"],  dtype=float)
    AC  = np.array(ds["AC"],      dtype=float)

    id2order = {o["id"]: o for o in orders}
    Pb = np.array([float(sum(id2order[j]["proc_body"]))     for j in ids])
    Pe = np.array([float(sum(id2order[j]["proc_cabinet"]))  for j in ids])
    Pp = np.array([float(id2order[j]["proc_pipe"])          for j in ids])

    w = prof - AC * (Pb + Pe + Pp)  # 净利润
    return ids, r, d, Pb, Pe, Pp, w

# ========== 三机解码（选择或跳过） ==========
def decode_schedule_by_perm_3m(perm_idx: List[int], ids: List[int],
                               r: np.ndarray, d: np.ndarray,
                               Pb: np.ndarray, Pe: np.ndarray, Pp: np.ndarray,
                               w:  np.ndarray,
                               accept_negative: bool=False):
    tb = te = tp = 0.0
    obj = 0.0
    selected, sched = [], {}

    # 记录用于“关键作业”识别的松弛度（未入选则为 +inf）
    slack = np.full(len(ids), np.inf)

    for pos, idx in enumerate(perm_idx):
        j = ids[idx]
        Sb = max(tb, r[idx]); Cb = Sb + Pb[idx]
        Se = max(te, r[idx]); Ce = Se + Pe[idx]
        Sp = max(tp, max(Cb, Ce)); Cp = Sp + Pp[idx]

        feas = (Cp <= d[idx]) and (accept_negative or w[idx] > 0.0)
        if feas:
            selected.append(j)
            sched[j] = (Sb, Cb, Se, Ce, Sp, Cp)
            tb, te, tp = Cb, Ce, Cp
            obj += w[idx]
            # 松弛度：离交期还有多少
            slack[idx] = d[idx] - Cp
        else:
            slack[idx] = np.inf  # 未被接纳

    return obj, selected, sched, slack

# ========== 邻域（含关键作业导向） ==========
def neighbor_perm(perm: np.ndarray, rng: np.random.Generator,
                  ids, r, d, Pb, Pe, Pp, w,
                  bias_critical: bool = True):
    """
    随机邻域 + 可选‘关键作业’导向：
      - 50% 传统：swap / insert / reverse
      - 50% 关键：把“松弛度小的作业”向前插（需要一次快速近似 slack）
    """
    n = len(perm)
    if n <= 2 or (not bias_critical):
        return _random_move(perm, rng)

    if rng.random() < 0.5:
        return _random_move(perm, rng)

    # —— 关键作业导向（近似 slack：用单机近似或一次粗解码）——
    # 为了低开销，这里用简单近似：用总时长替代三机，估算一个 “紧迫度分数”
    totP = Pb + Pe + Pp
    urgency = np.maximum(0.0, r + totP - d) + (1.0 / (1.0 + totP))  # 越大越紧
    # 在当前 perm 上，从后半段选择一个更可能“来不及”的作业
    tail_start = n // 2
    cand_pos = rng.integers(tail_start, n)
    idx = perm[cand_pos]
    # 向前插到一个随机的较靠前位置（或更靠近释放期的区间）
    target = rng.integers(0, max(1, cand_pos))
    newp = perm.copy()
    # 执行 insert
    newp = np.delete(newp, cand_pos)
    newp = np.insert(newp, target, idx)
    return newp

def _random_move(perm: np.ndarray, rng: np.random.Generator):
    n = len(perm)
    i, j = rng.integers(0, n, size=2)
    if i == j: j = (j + 1) % n
    move = rng.choice(["swap", "insert", "reverse"])
    newp = perm.copy()
    if move == "swap":
        newp[i], newp[j] = newp[j], newp[i]
    elif move == "insert":
        a, b = (i, j) if i < j else (j, i)
        val = newp[a]
        newp = np.delete(newp, a)
        newp = np.insert(newp, b, val)
    else:
        a, b = (i, j) if i < j else (j, i)
        newp[a:b+1] = newp[a:b+1][::-1]
    return newp

# ========== 初温自动标定 ==========
def auto_calibrate_T0(curr_perm: np.ndarray, ids, r, d, Pb, Pe, Pp, w,
                      rng: np.random.Generator, samples: int = 60, p0: float = 0.8):
    """
    估计初期的‘负增量’平均幅度，然后设 T0 使得坏解的接受概率 ~ p0
      p0 ≈ exp( - |Δ| / T0 )  →  T0 ≈ mean_neg_delta / ln(1/p0)
    """
    # 当前解的 obj
    curr_obj, _, _, _ = decode_schedule_by_perm_3m(curr_perm.tolist(), ids, r, d, Pb, Pe, Pp, w, accept_negative=False)
    negs = []
    for _ in range(samples):
        cand_perm = _random_move(curr_perm, rng)
        cand_obj, _, _, _ = decode_schedule_by_perm_3m(cand_perm.tolist(), ids, r, d, Pb, Pe, Pp, w, accept_negative=False)
        delta = cand_obj - curr_obj
        if delta < 0:
            negs.append(-delta)
    if len(negs) == 0:
        return 1.0  # 极少见：当前就很难变差，给个保底
    mean_neg = float(np.mean(negs))
    T0 = mean_neg / max(1e-9, np.log(1.0 / max(1e-4, min(0.9999, p0))))
    return T0

# ========== 模拟退火主过程 ==========
def solve_selection_sa_3m(ds: Dict,
                          T0: float = None,        # 若为 None 则自动标定
                          Tmin: float = 1e-3,
                          alpha: float = 0.97,
                          iters_per_T: int = None, # 若为 None 则用 20*n
                          accept_negative: bool = False,
                          seed: int = 2025,
                          log_every: int = 500,
                          reheats: int = 2):       # 回火次数上限
    rng = np.random.default_rng(seed)
    ids, r, d, Pb, Pe, Pp, w = build_data_3m(ds)
    n = len(ids)
    if n == 0:
        return {"selected_ids": [], "schedule": {}, "obj": 0.0, "best_perm": []}

    if iters_per_T is None:
        iters_per_T = max(200, 20 * n)  # 与规模挂钩

    # 初始解：单位时间收益降序
    ratio = np.where(Pb + Pe + Pp > 0, w / (Pb + Pe + Pp), -1e9)
    curr_perm = np.argsort(-ratio, kind="mergesort")
    curr_obj, _, _, _ = decode_schedule_by_perm_3m(curr_perm.tolist(), ids, r, d, Pb, Pe, Pp, w, accept_negative)
    best_perm = curr_perm.copy()
    best_obj  = curr_obj

    # 自动标定初温（关键！）
    if T0 is None:
        T0 = auto_calibrate_T0(curr_perm, ids, r, d, Pb, Pe, Pp, w, rng, samples=80, p0=0.80)

    T = float(T0)
    no_improve_levels = 0
    used_reheats = 0
    global_iter = 0

    while T > Tmin:
        improved_this_level = False
        for _ in range(iters_per_T):
            global_iter += 1
            cand_perm = neighbor_perm(curr_perm, rng, ids, r, d, Pb, Pe, Pp, w, bias_critical=True)
            cand_obj, _, _, _ = decode_schedule_by_perm_3m(cand_perm.tolist(), ids, r, d, Pb, Pe, Pp, w, accept_negative)
            delta = cand_obj - curr_obj

            # 退火接受
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                curr_perm, curr_obj = cand_perm, cand_obj
                improved_this_level = improved_this_level or (delta > 1e-9)

                # 小型局部强化：把1~2个“临期作业”向前再插一下
                if delta > 0:
                    for _ in range(2):
                        curr_perm = _random_move(curr_perm, rng)  # 便宜的微调
                        tmp_obj, _, _, _ = decode_schedule_by_perm_3m(curr_perm.tolist(), ids, r, d, Pb, Pe, Pp, w, accept_negative)
                        if tmp_obj >= curr_obj:
                            curr_obj = tmp_obj
                        else:
                            # 回退
                            pass

                if curr_obj > best_obj + 1e-9:
                    best_perm = curr_perm.copy()
                    best_obj  = curr_obj

            if (log_every is not None) and (global_iter % log_every == 0):
                print(f"[SA-3M] step={global_iter:6d}  T={T:.3f}  best={best_obj:.2f}")

        # 降温 / 回火策略
        if improved_this_level:
            no_improve_levels = 0
        else:
            no_improve_levels += 1

        if no_improve_levels >= 2 and used_reheats < reheats:
            # 回火：提高温度，打破平台；并稍微增大每层迭代
            T *= 1.0 / max(1e-6, alpha)  # 提高到上一层温度
            used_reheats += 1
            iters_per_T = int(iters_per_T * 1.2)
            no_improve_levels = 0
            # 也可微扰当前解
            curr_perm = _random_move(curr_perm, rng)
            curr_obj, _, _, _ = decode_schedule_by_perm_3m(curr_perm.tolist(), ids, r, d, Pb, Pe, Pp, w, accept_negative)
        else:
            # 正常降温
            T *= alpha

    # 输出最优解
    _, best_selected, best_sched, _ = decode_schedule_by_perm_3m(best_perm.tolist(), ids, r, d, Pb, Pe, Pp, w, accept_negative)
    return {
        "selected_ids": best_selected,
        "schedule": best_sched,
        "obj": best_obj,
        "best_perm": best_perm.tolist()
    }

# ========== 运行示例 ==========
if __name__ == "__main__":
    res = solve_selection_sa_3m(selection_ds,
                                T0=None,          # 让它自动标定
                                Tmin=1e-3,
                                alpha=0.97,
                                iters_per_T=None, # 让它用 20*n
                                accept_negative=False,
                                seed=42,
                                log_every=500,
                                reheats=2)
    print("\n[SA-3M] 选中的特殊订单ID：", sorted(res["selected_ids"]))
    seq = sorted(res["selected_ids"], key=lambda j: res["schedule"][j][4])
    print("[SA-3M] 加工顺序：")
    for pos, j in enumerate(seq, 1):
        Sb,Cb, Se,Ce, Sp,Cp = res["schedule"][j]
        idx = selection_ds["ids"].index(j)
        rj, dj = selection_ds["release"][idx], selection_ds["due"][idx]
        print(f"{pos:02d}. ID={j} | r={rj:.0f}, d={dj:.0f} | "
              f"Sb={Sb:.1f},Cb={Cb:.1f}, Se={Se:.1f},Ce={Ce:.1f}, Sp={Sp:.1f},Cp={Cp:.1f}")
    print(f"[SA-3M] 目标(净利润) = {res['obj']:.2f}")
