# selection_model_gurobi.py
from gurobipy import Model, GRB, quicksum
from parameter import selection_ds, big_M   #调用Big-M与订单数据

def _get_M_for_selection(big_M_obj):
    """从 parameter.big_M 推导选择模型的 Big-M（标量）"""
    try:
        # 若是 dict: {"body": Mb, "cabinet": Me, "pipe": Mp}
        Mb = float(big_M_obj.get("body", 0.0))
        Me = float(big_M_obj.get("cabinet", 0.0))
        Mp = float(big_M_obj.get("pipe", 0.0))
        return max(Mb, Me) + Mp
    except AttributeError:
        # 若是标量
        return float(big_M_obj)

# Big-M：直接用 parameter.big_M
M = _get_M_for_selection(big_M)

def solve_selection_gurobi(time_limit, log=True):
    # 读取数据（只用特殊订单）
    ids     = list(selection_ds["ids"])
    release = {j: float(r) for j, r in zip(ids, selection_ds["release"])}
    due     = {j: float(d) for j, d in zip(ids, selection_ds["due"])}
    ptilde  = {j: float(p) for j, p in zip(ids, selection_ds["p_tilde"])}
    profit  = {j: float(v) for j, v in zip(ids, selection_ds["profit"])}
    AC      = {j: float(v) for j, v in zip(ids, selection_ds["AC"])}

    if "net_profit" in selection_ds:
        w = {j: float(v) for j, v in zip(ids, selection_ds["net_profit"])}
    else:
        w = {j: profit[j] - AC[j]*ptilde[j] for j in ids}

    # 建模
    m = Model("dominant_job_selection")
    if not log:
        m.Params.LogToConsole = 0
    m.Params.TimeLimit = time_limit

    # 变量
    y = {j: m.addVar(vtype=GRB.BINARY, name=f"y[{j}]") for j in ids}
    # 规划水平（仅用于变量上界，不影响可行性）
    H = max(due.values()) + sum(ptilde.values()) if ids else 0.0
    S = {j: m.addVar(lb=0.0, ub=H, name=f"S[{j}]") for j in ids}
    C = {j: m.addVar(lb=0.0, ub=H, name=f"C[{j}]") for j in ids}
    Z = {}
    for i in ids:
        for j in ids:
            if i != j:
                Z[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"Z[{i},{j}]")
    m.update()

    # (2) 释放期
    for j in ids:
        m.addConstr(S[j] >= release[j] * y[j], name=f"rel[{j}]")

    # (3) 时长门控
    for j in ids:
        m.addConstr(C[j] >= S[j] + ptilde[j]*y[j], name=f"dur[{j}]")

    # 硬交期（不迟交）：C <= d + M(1 - y)
    for j in ids:
        m.addConstr(due[j] >= C[j] + M*(y[j] - 1), name=f"due[{j}]")

    # (4) 序约束 with Big-M：S_j >= C_i - M(1 - Z_ij)
    for i in ids:
        for j in ids:
            if i == j:
                continue
            m.addConstr(S[j] >= C[i] - M*(1 - Z[(i, j)]), name=f"seq[{i}->{j}]")

    # (5)(6) 门控：使用任何 Z 即必须被选择
    for i in ids:
        for j in ids:
            if i == j:
                continue
            m.addConstr(y[j] >= Z[(i, j)] + Z[(j, i)], name=f"gate_j[{i},{j}]")
            m.addConstr(y[i] >= Z[(i, j)] + Z[(j, i)], name=f"gate_i[{i},{j}]")

    # (7) 闭合：“二选一”
    for i_idx, i in enumerate(ids):
        for j in ids[i_idx+1:]:
            m.addConstr(Z[(i, j)] + Z[(j, i)] >= y[i] + y[j] - 1, name=f"close[{i},{j}]")

    # 目标：Max TP = sum_j y_j * net_profit_j
    m.setObjective(quicksum(w[j]*y[j] for j in ids), GRB.MAXIMIZE)

    m.optimize()

    sel = []
    sched = {}
    if m.SolCount > 0:
        for j in ids:
            if y[j].X > 0.5:
                sel.append(j)
                sched[j] = (float(S[j].X), float(C[j].X), float(release[j]), float(due[j]))
    return {"selected_ids": sel, "schedule": sched, "obj": (m.ObjVal if m.SolCount>0 else None)}

# 直接运行示例（可删）
if __name__ == "__main__":
    res = solve_selection_gurobi(time_limit=1200, log=True)
    print("选中的特殊订单ID：", sorted(res["selected_ids"]))
    for j, (s, c, r, d) in list(res["schedule"].items())[:5]:
        print(f"  ID={j}, S={s:.1f}, C={c:.1f}, R={r:.1f}, Due = {d:.1f}")
    print("TP =", res["obj"])
