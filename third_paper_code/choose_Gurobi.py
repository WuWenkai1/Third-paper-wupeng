# choose_model_3machines.py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from gurobipy import Model, GRB, quicksum
from parameter import selection_ds, big_M    # 必需
# 若 parameter 暴露了 orders（建议）就一起导入：
try:
    from parameter import orders
except Exception:
    orders = None

# ------------------------------------------------------------
# 1) 读取并整理数据（只含“特殊订单”）
# ------------------------------------------------------------
ids = list(selection_ds["ids"])
r   = {j: float(selection_ds["release"][k])     for k, j in enumerate(ids)}
d   = {j: float(selection_ds["due"][k])         for k, j in enumerate(ids)}
AC  = {j: float(selection_ds["AC"][k])          for k, j in enumerate(ids)}
prof= {j: float(selection_ds["profit"][k])      for k, j in enumerate(ids)}

# 尝试从 parameter.orders 中提取三条线的加工时间（累加本体/电柜工序）
def _extract_processing_from_orders(ids, orders):
    # 返回三条线的加工时间字典
    Pb, Pe, Pp = {}, {}, {}
    id2order = {o["id"]: o for o in orders}
    for j in ids:
        o = id2order[j]
        Pb[j] = float(sum(o["proc_body"]))       # 本体3道工序之和
        Pe[j] = float(sum(o["proc_cabinet"]))    # 电柜5道工序之和
        Pp[j] = float(o["proc_pipe"])            # 装配单工序
    return Pb, Pe, Pp

if orders is None:
    raise RuntimeError(
        "需要从 parameter 导入 orders 才能拿到各订单在三条线的加工时间。"
        "请在 parameter.py 里确保 `orders` 变量可被导入。"
    )
Pb, Pe, Pp = _extract_processing_from_orders(ids, orders)

# Big-M：按产线分别使用（来自 parameter.big_M）
Mb = float(big_M.get("body",    sum(Pb.values())))
Me = float(big_M.get("cabinet", sum(Pe.values())))
Mp = float(big_M.get("pipe",    sum(Pp.values())))

# ------------------------------------------------------------
# 2) 建模
# ------------------------------------------------------------
m = Model("Choose_3Machines_MaxProfit")
m.Params.OutputFlag = 1  # 控制台打印日志（=0关闭）

# --- 决策变量
y = {j: m.addVar(vtype=GRB.BINARY, name=f"y[{j}]") for j in ids}

Sb = {j: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Sb[{j}]") for j in ids}
Cb = {j: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Cb[{j}]") for j in ids}

Se = {j: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Se[{j}]") for j in ids}
Ce = {j: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Ce[{j}]") for j in ids}

Sp = {j: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Sp[{j}]") for j in ids}
Cp = {j: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Cp[{j}]") for j in ids}

# 先后变量：对每条线分别建 X^ell_{ij}（有向对 i!=j）
Xb = {}
Xe = {}
Xp = {}
for i in ids:
    for j in ids:
        if i == j:
            continue
        Xb[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"Xb[{i},{j}]")
        Xe[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"Xe[{i},{j}]")
        Xp[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"Xp[{i},{j}]")

m.update()

# ------------------------------------------------------------
# 3) 约束
# ------------------------------------------------------------

# 3.1 释放期（b/e 产线）
for j in ids:
    m.addConstr(Sb[j] >= r[j] * y[j], name=f"release_b[{j}]")
    m.addConstr(Se[j] >= r[j] * y[j], name=f"release_e[{j}]")

# 3.2 同线加工持续时间
for j in ids:
    m.addConstr(Cb[j] >= Sb[j] + Pb[j] * y[j], name=f"dur_b[{j}]")
    m.addConstr(Ce[j] >= Se[j] + Pe[j] * y[j], name=f"dur_e[{j}]")
    m.addConstr(Cp[j] >= Sp[j] + Pp[j] * y[j], name=f"dur_p[{j}]")

# 3.3 装配汇合：必须等 b/e 完成（用 Big-M 门控，y=0 时放松）
for j in ids:
    m.addConstr(Sp[j] >= Cb[j] + (y[j] - 1.0) * Mb, name=f"join_b[{j}]")
    m.addConstr(Sp[j] >= Ce[j] + (y[j] - 1.0) * Me, name=f"join_e[{j}]")

# 3.4 截止期（只对被接单的生效）
for j in ids:
    m.addConstr(Cp[j] <= d[j] + (1.0 - y[j]) * Mp, name=f"due[{j}]")

# 3.5 不重叠（3 条线分别建），含“二选一 + 门控”
def add_no_overlap(X, S, C, M, line_tag):
    # X[(i,j)] = 1 表示 i 在 j 之前
    for i in ids:
        for j in ids:
            if i == j:
                continue
            # 非重叠 big-M
            m.addConstr(S[i] >= C[j] - M * (1.0 - X[(j, i)]),
                        name=f"noovl_{line_tag}_({i})after({j})")
            m.addConstr(S[j] >= C[i] - M * (1.0 - X[(i, j)]),
                        name=f"noovl_{line_tag}_({j})after({i})")
            # 二选一 + 门控：若两单都被选，则必须二选一；若任一未选，变量可为 0
#            m.addConstr(X[(i, j)] + X[(j, i)] <= 1.0, name=f"xor1_{line_tag}[{i},{j}]")
            m.addConstr(X[(i, j)] + X[(j, i)] >= y[i] + y[j] - 1.0,
                        name=f"xor2_{line_tag}[{i},{j}]")
            # 门控（可选，增强数值稳定）
            m.addConstr(X[(i, j)] + X[(j, i)] <= y[i], name=f"gatei_{line_tag}[{i},{j}]")
            m.addConstr(X[(i, j)] + X[(j, i)] <= y[j], name=f"gatej_{line_tag}[{i},{j}]")

add_no_overlap(Xb, Sb, Cb, Mb, "b")
add_no_overlap(Xe, Se, Ce, Me, "e")
add_no_overlap(Xp, Sp, Cp, Mp, "p")

# ------------------------------------------------------------
# 4) 目标：最大化利润
#     TP = sum_j y_j * (profit_j - AC_j * (P^b_j + P^e_j + P^p_j))
# ------------------------------------------------------------
obj = quicksum(
    y[j] * (prof[j] - AC[j] * (Pb[j] + Pe[j] + Pp[j])) for j in ids
)
m.setObjective(obj, GRB.MAXIMIZE)

# ------------------------------------------------------------
# 7) 画甘特图（Body / Cabinet / Pipeline）
# ------------------------------------------------------------

def _collect_intervals(selected_ids):
    """把每条产线转换为 [(job, start, width)]，只画被选择的订单"""
    body = []
    cab  = []
    pipe = []
    for j in selected_ids:
        body.append((j, Sb[j].X, Cb[j].X - Sb[j].X))
        cab.append( (j, Se[j].X, Ce[j].X - Se[j].X))
        pipe.append((j, Sp[j].X, Cp[j].X - Sp[j].X))
    # 按开始时间排序，方便标注
    body.sort(key=lambda x: x[1])
    cab.sort(key=lambda x: x[1])
    pipe.sort(key=lambda x: x[1])
    return body, cab, pipe

def plot_gantt(selected_ids, save_path=None, figsize=(12, 4.5), title="Schedule Gantt"):
    body, cab, pipe = _collect_intervals(selected_ids)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    # y 轴三条产线的“行”位置
    y_body, y_cab, y_pipe = 30, 20, 10           # 三条横线的 y 位置
    h = 6                                        # 每条横线的高度（bar 厚度）

    # 为不同产线定义颜色
    color_b = "#377eb8"
    color_e = "#4daf4a"
    color_p = "#e41a1c"

    # 画 body
    for (j, s, w) in body:
        ax.broken_barh([(s, w)], (y_body, h), facecolors=color_b, edgecolor="black", alpha=0.9)
        ax.text(s + w/2, y_body + h/2, f"{j}", ha="center", va="center", fontsize=8, color="white")

    # 画 cabinet
    for (j, s, w) in cab:
        ax.broken_barh([(s, w)], (y_cab, h), facecolors=color_e, edgecolor="black", alpha=0.9)
        ax.text(s + w/2, y_cab + h/2, f"{j}", ha="center", va="center", fontsize=8, color="white")

    # 画 pipeline
    for (j, s, w) in pipe:
        ax.broken_barh([(s, w)], (y_pipe, h), facecolors=color_p, edgecolor="black", alpha=0.9)
        ax.text(s + w/2, y_pipe + h/2, f"{j}", ha="center", va="center", fontsize=8, color="white")

    # 样式
    ax.set_yticks([y_pipe + h/2, y_cab + h/2, y_body + h/2])
    ax.set_yticklabels(["Pipeline", "Cabinet", "Body"], fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_title(title, fontsize=12)
    # x 轴范围到总体 makespan
    try:
        makespan = max(Cp[j].X for j in selected_ids) if selected_ids else 0.0
    except Exception:
        makespan = 0.0
    ax.set_xlim(left=0, right=max(makespan * 1.05, 1.0))
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # 图例
    legend_elems = [
        Patch(facecolor=color_b, edgecolor="black", label="Body"),
        Patch(facecolor=color_e, edgecolor="black", label="Cabinet"),
        Patch(facecolor=color_p, edgecolor="black", label="Pipeline"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Gantt saved to: {save_path}")
    plt.show()

# ------------------------------------------------------------
# 5) 求解
# ------------------------------------------------------------
# 一些稳健参数（可按需调整）
m.Params.Presolve   = 2
m.Params.Cuts       = 2
m.Params.MIPFocus   = 1     # 先找好解
m.Params.Heuristics = 0.2
m.Params.TimeLimit  = 2400  # 如需限时可开启

m.optimize()

# ------------------------------------------------------------
# 6) 输出结果
# ------------------------------------------------------------
if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT or m.Status == GRB.INTERRUPTED:
    sel = [j for j in ids if y[j].X > 0.5]
    plot_gantt(sel, save_path="gantt_selection.png",
               title=f"Gantt (|sel|={len(sel)}, Profit={m.ObjVal:.0f})")
    print("\n== 被选择的特殊订单ID（升序） ==")
    print(sorted(sel))

    # 按装配开始时间给出顺序（更贴近出货次序）
    seq = sorted(sel, key=lambda j: Sp[j].X)
    print("\n== 选择结果的装配顺序（S_p 升序） ==")
    for k, j in enumerate(seq, 1):
        print(f"{k:02d}. ID={j} | r={r[j]:.0f}, d={d[j]:.0f} | "
              f"Sb={Sb[j].X:.1f}, Cb={Cb[j].X:.1f} ; "
              f"Se={Se[j].X:.1f}, Ce={Ce[j].X:.1f} ; "
              f"Sp={Sp[j].X:.1f}, Cp={Cp[j].X:.1f}")

    print(f"\n目标（总利润）= {m.ObjVal:.2f}")
else:
    print(f"模型未得到可行解，Gurobi 状态码 = {m.Status}")
