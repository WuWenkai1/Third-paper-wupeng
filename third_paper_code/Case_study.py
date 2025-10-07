# -*- coding: utf-8 -*-
from gurobipy import Model, GRB, quicksum
import itertools, math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec, rcParams
from matplotlib.ticker import MultipleLocator

# ------------ Data ------------
J_SPECIAL = ["D1","D2","D3"]
J_NORMAL  = ["N1","N2","N3","N4"]
JOBS = J_SPECIAL + J_NORMAL

P_BODY = {
    "D1":[5,3,5], "D2":[3,5,3], "D3":[2,3,2],
    "N1":[5,2,4], "N2":[4,3,4], "N3":[2,5,3], "N4":[5,4,2],
}
P_ECAB = {
    "D1":[1,3,2,4,5], "D2":[2,3,2,3,2], "D3":[1,3,2,1,3],
    "N1":[2,4,2,4,1], "N2":[5,1,2,3,1], "N3":[2,4,2,1,4], "N4":[4,3,1,1,3],
}
P_PACK = {"D1":3,"D2":4,"D3":5,"N1":3,"N2":2,"N3":3,"N4":2}

R   = {"D1":5,"D2":20,"D3":25,"N1":0,"N2":0,"N3":0,"N4":0}
DUE = {"D1":25,"D2":40,"D3":43,"N1":25,"N2":20,"N3":15,"N4":25}

PI  = {"D1":39000,"D2":30000,"D3":27000,"N1":13500,"N2":12400,"N3":13300,"N4":15900}
U   = {"N1":20,"N2":20,"N3":25,"N4":20}
AC  = {"D1":500,"D2":442,"D3":403,"N1":318,"N2":275,"N3":294,"N4":272}

ALPHAS = [0.3,0.4,0.5,0.6,0.7]

def total_base_time(j):
    return sum(P_BODY[j]) + sum(P_ECAB[j]) + P_PACK[j]

H = sum(total_base_time(j) for j in JOBS) + max(DUE.values()) + max(R.values())
Mbig = H

# ---------- helper: ready times ----------
def add_ready_defs(m, s_b, c_b, s_e, c_e, s_p, c_p):
    # E on body/ecab: first stage -> release; later -> prev completion
    E_b = {(p,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"E_b[{p},{j}]")
           for p in range(3) for j in JOBS}
    E_e = {(q,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"E_e[{q},{j}]")
           for q in range(5) for j in JOBS}
    for j in JOBS:
        m.addConstr(E_b[(0,j)] == R[j])
        for p in range(1,3):
            m.addConstr(E_b[(p,j)] == c_b[(p-1,j)])
        m.addConstr(E_e[(0,j)] == R[j])
        for q in range(1,5):
            m.addConstr(E_e[(q,j)] == c_e[(q-1,j)])
    # PACK ready = max of two last-stage completions
    E_p = {j: m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"E_p[{j}]") for j in JOBS}
    th  = {j: m.addVar(vtype=GRB.BINARY, name=f"th[{j}]") for j in JOBS}
    for j in JOBS:
        m.addConstr(E_p[j] >= c_b[(2,j)])
        m.addConstr(E_p[j] >= c_e[(4,j)])
        m.addConstr(E_p[j] <= c_b[(2,j)] + Mbig*th[j])
        m.addConstr(E_p[j] <= c_e[(4,j)] + Mbig*(1 - th[j]))
    return E_b, E_e, E_p

# ---------- build model with immediate-predecessor y ----------
def build_model_soft_noidle():
    m = Model()

    # 1) 变量：起止时间
    s_b = {(p,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"s_b[{p},{j}]")
           for p in range(3) for j in JOBS}
    c_b = {(p,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"c_b[{p},{j}]")
           for p in range(3) for j in JOBS}
    s_e = {(q,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"s_e[{q},{j}]")
           for q in range(5) for j in JOBS}
    c_e = {(q,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"c_e[{q},{j}]")
           for q in range(5) for j in JOBS}
    s_p = {j: m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"s_p[{j}]") for j in JOBS}
    c_p = {j: m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"c_p[{j}]") for j in JOBS}

    # 完成定义
    for p in range(3):
        for j in JOBS:
            m.addConstr(c_b[(p,j)] == s_b[(p,j)] + P_BODY[j][p])
    for q in range(5):
        for j in JOBS:
            m.addConstr(c_e[(q,j)] == s_e[(q,j)] + P_ECAB[j][q])
    for j in JOBS:
        m.addConstr(c_p[j] == s_p[j] + P_PACK[j])

    # 2) 就绪时间（与之前一致）
    E_b = {(p,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"E_b[{p},{j}]")
           for p in range(3) for j in JOBS}
    E_e = {(q,j): m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"E_e[{q},{j}]")
           for q in range(5) for j in JOBS}
    for j in JOBS:
        m.addConstr(E_b[(0,j)] == R[j])
        for p in range(1,3):
            m.addConstr(E_b[(p,j)] == c_b[(p-1,j)])
        m.addConstr(E_e[(0,j)] == R[j])
        for q in range(1,5):
            m.addConstr(E_e[(q,j)] == c_e[(q-1,j)])

    E_p = {j: m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"E_p[{j}]") for j in JOBS}
    th  = {j: m.addVar(vtype=GRB.BINARY, name=f"th[{j}]") for j in JOBS}
    for j in JOBS:
        m.addConstr(E_p[j] >= c_b[(2,j)])
        m.addConstr(E_p[j] >= c_e[(4,j)])
        m.addConstr(E_p[j] <= c_b[(2,j)] + Mbig*th[j])
        m.addConstr(E_p[j] <= c_e[(4,j)] + Mbig*(1 - th[j]))

    # 3) 邻接变量（源/汇点）——只保留“度约束”，不写上界等式
    SRC, SNK = "SRC", "SNK"
    nodes = JOBS + [SRC, SNK]

    y_b = {(p,i,j): m.addVar(vtype=GRB.BINARY, name=f"y_b[{p},{i},{j}]")
           for p in range(3) for i in nodes for j in nodes if i != j}
    y_e = {(q,i,j): m.addVar(vtype=GRB.BINARY, name=f"y_e[{q},{i},{j}]")
           for q in range(5) for i in nodes for j in nodes if i != j}
    y_p = {(i,j):   m.addVar(vtype=GRB.BINARY, name=f"y_p[{i},{j}]")
           for i in nodes for j in nodes if i != j}

    # 度约束（与之前相同）
    for p in range(3):
        for j in JOBS:
            m.addConstr(quicksum(y_b[(p,i,j)] for i in nodes if i!=j) == 1)
            m.addConstr(quicksum(y_b[(p,j,k)] for k in nodes if k!=j) == 1)
        m.addConstr(quicksum(y_b[(p,SRC,k)] for k in JOBS) == 1)
        m.addConstr(quicksum(y_b[(p,k,SNK)] for k in JOBS) == 1)
        m.addConstr(quicksum(y_b[(p,i,SRC)] for i in nodes if i!=SRC) == 0)
        m.addConstr(quicksum(y_b[(p,SNK,k)] for k in nodes if k!=SNK) == 0)

    for q in range(5):
        for j in JOBS:
            m.addConstr(quicksum(y_e[(q,i,j)] for i in nodes if i!=j) == 1)
            m.addConstr(quicksum(y_e[(q,j,k)] for k in nodes if k!=j) == 1)
        m.addConstr(quicksum(y_e[(q,SRC,k)] for k in JOBS) == 1)
        m.addConstr(quicksum(y_e[(q,k,SNK)] for k in JOBS) == 1)
        m.addConstr(quicksum(y_e[(q,i,SRC)] for i in nodes if i!=SRC) == 0)
        m.addConstr(quicksum(y_e[(q,SNK,k)] for k in nodes if k!=SNK) == 0)

    for j in JOBS:
        m.addConstr(quicksum(y_p[(i,j)] for i in nodes if i!=j) == 1)
        m.addConstr(quicksum(y_p[(j,k)] for k in nodes if k!=j) == 1)
    m.addConstr(quicksum(y_p[(SRC,k)] for k in JOBS) == 1)
    m.addConstr(quicksum(y_p[(k,SNK)] for k in JOBS) == 1)
    m.addConstr(quicksum(y_p[(i,SRC)] for i in nodes if i!=SRC) == 0)
    m.addConstr(quicksum(y_p[(SNK,k)] for k in nodes if k!=SNK) == 0)

    # 4) 仅保留下界（防重叠 + 就绪），并引入“空闲量”变量
    idle_terms = []

    # BODY
    for p in range(3):
        for i in JOBS:
            for j in JOBS:
                if i == j: continue
                # s_j >= c_i 和 s_j >= E_j （当 i 紧邻 j）
                m.addConstr(s_b[(p,j)] >= c_b[(p,i)] - Mbig*(1 - y_b[(p,i,j)]))
                m.addConstr(s_b[(p,j)] >= E_b[(p,j)] - Mbig*(1 - y_b[(p,i,j)]))
        # 源点：s_j >= E_j
        for j in JOBS:
            m.addConstr(s_b[(p,j)] >= E_b[(p,j)] - Mbig*(1 - y_b[(p,SRC,j)]))
        # 空闲量：s_j - max(...) 用一个松弛（>=0）来表示并累加最小化
        # 用两条约束把 idle >= s_j - c_i  和 idle >= s_j - E_j
        for i in nodes:
            for j in JOBS:
                if i == j or i == SNK: continue
                idle = m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"id_b[{p},{i},{j}]")
                idle_terms.append(idle)
                # 当 y=1 时才有效
                if i != SRC:
                    m.addConstr(idle >= s_b[(p,j)] - c_b[(p,i)] - Mbig*(1 - y_b[(p,i,j)]))
                m.addConstr(idle >= s_b[(p,j)] - E_b[(p,j)] - Mbig*(1 - y_b[(p,i,j)]))

    # ECAB（同理）
    for q in range(5):
        for i in JOBS:
            for j in JOBS:
                if i == j: continue
                m.addConstr(s_e[(q,j)] >= c_e[(q,i)] - Mbig*(1 - y_e[(q,i,j)]))
                m.addConstr(s_e[(q,j)] >= E_e[(q,j)] - Mbig*(1 - y_e[(q,i,j)]))
        for j in JOBS:
            m.addConstr(s_e[(q,j)] >= E_e[(q,j)] - Mbig*(1 - y_e[(q,SRC,j)]))
        for i in nodes:
            for j in JOBS:
                if i == j or i == SNK: continue
                idle = m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"id_e[{q},{i},{j}]")
                idle_terms.append(idle)
                if i != SRC:
                    m.addConstr(idle >= s_e[(q,j)] - c_e[(q,i)] - Mbig*(1 - y_e[(q,i,j)]))
                m.addConstr(idle >= s_e[(q,j)] - E_e[(q,j)] - Mbig*(1 - y_e[(q,i,j)]))

    # PACK
    for i in JOBS:
        for j in JOBS:
            if i == j: continue
            m.addConstr(s_p[j] >= c_p[i] - Mbig*(1 - y_p[(i,j)]))
            m.addConstr(s_p[j] >= E_p[j] - Mbig*(1 - y_p[(i,j)]))
    for j in JOBS:
        m.addConstr(s_p[j] >= E_p[j] - Mbig*(1 - y_p[(SRC,j)]))
    for i in nodes:
        for j in JOBS:
            if i == j or i == SNK: continue
            idle = m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"id_p[{i},{j}]")
            idle_terms.append(idle)
            if i != SRC:
                m.addConstr(idle >= s_p[j] - c_p[i] - Mbig*(1 - y_p[(i,j)]))
            m.addConstr(idle >= s_p[j] - E_p[j] - Mbig*(1 - y_p[(i,j)]))

    # 5) 跨工位的工序顺序（同你原来）
    for j in JOBS:
        m.addConstr(s_b[(1,j)] >= c_b[(0,j)])
        m.addConstr(s_b[(2,j)] >= c_b[(1,j)])
        m.addConstr(s_e[(1,j)] >= c_e[(0,j)])
        m.addConstr(s_e[(2,j)] >= c_e[(1,j)])
        m.addConstr(s_e[(3,j)] >= c_e[(2,j)])
        m.addConstr(s_e[(4,j)] >= c_e[(3,j)])
        m.addConstr(s_p[j] >= c_b[(2,j)])
        m.addConstr(s_p[j] >= c_e[(4,j)])

    # 特殊订单按时
    for j in J_SPECIAL:
        m.addConstr(c_p[j] <= DUE[j])

    return m, s_b, c_b, s_e, c_e, s_p, c_p, idle_terms

# ---------- evaluation ----------
def add_eval_terms(m, s_p, c_p, alpha, tag):
    Tn   = {j: m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"T[{j}]_{tag}") for j in J_NORMAL}
    Cmax = m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"Cmax_{tag}")
    for j in J_NORMAL:
        m.addConstr(Tn[j] >= c_p[j] - DUE[j]); m.addConstr(Tn[j] >= 0)
    for j in JOBS:
        m.addConstr(Cmax >= c_p[j])
    revenue = quicksum(PI[j] for j in JOBS)
    mfgcost = quicksum(AC[j]*(sum(P_BODY[j])+sum(P_ECAB[j])+P_PACK[j]) for j in JOBS)
    tard    = quicksum(U[j]*Tn[j] for j in J_NORMAL)
    TP = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"TP_{tag}")
    f  = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_{tag}")
    m.addConstr(TP == revenue - mfgcost - tard)
    m.addConstr(f  == alpha*TP - (1 - alpha)*1000*Cmax)
    return {"Tn":Tn, "Cmax":Cmax, "TP":TP, "f":f}

# ---------- Phase 1 ----------
EPS_IDLE = 1e-6  # 极小系数

def compute_T_alpha05():
    m, s_b, c_b, s_e, c_e, s_p, c_p, idle_terms = build_model_soft_noidle()
    ev = add_eval_terms(m, s_p, c_p, alpha=0.5, tag="a0_5")
    m.setObjective(ev["f"] - EPS_IDLE*quicksum(idle_terms), GRB.MAXIMIZE)
    m.Params.MIPGap = 1e-3; m.Params.TimeLimit = 1200
    m.optimize()
    if m.Status == GRB.INFEASIBLE:
        m.computeIIS(); m.write("phase1_iis.ilp")
        raise RuntimeError("Phase-1 infeasible")
    sol = {
        "T": ev["f"].X, "TP": ev["TP"].X, "Cmax": ev["Cmax"].X,
        "s_b": {(p,j): s_b[(p,j)].X for p in range(3) for j in JOBS},
        "c_b": {(p,j): c_b[(p,j)].X for p in range(3) for j in JOBS},
        "s_e": {(q,j): s_e[(q,j)].X for q in range(5) for j in JOBS},
        "c_e": {(q,j): c_e[(q,j)].X for q in range(5) for j in JOBS},
        "s_p": {j: s_p[j].X for j in JOBS},
        "c_p": {j: c_p[j].X for j in JOBS},
        "order_pack": sorted(JOBS, key=lambda jj: s_p[jj].X)
    }
    return sol

# ---------- Phase 2 ----------
def solve_min_sum_PT(T_value, alphas=ALPHAS):
    # 1) 同一套排程变量 + 软无空隙
    m, s_b, c_b, s_e, c_e, s_p, c_p, idle_terms = build_model_soft_noidle()

    # 2) ——关键改动：只建一套 Tn/Cmax/TP（所有 α 共用）——
    Tn = {j: m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name=f"T[{j}]") for j in J_NORMAL}
    Cmax = m.addVar(lb=0, ub=H, vtype=GRB.CONTINUOUS, name="Cmax")
    for j in J_NORMAL:
        m.addConstr(Tn[j] >= c_p[j] - DUE[j])
        m.addConstr(Tn[j] >= 0)
    for j in JOBS:
        m.addConstr(Cmax >= c_p[j])

    # 经济项（常数表达式 + 变量 TP）
    revenue = quicksum(PI[j] for j in JOBS)
    mfgcost = quicksum(AC[j] * (sum(P_BODY[j]) + sum(P_ECAB[j]) + P_PACK[j]) for j in JOBS)
    tard    = quicksum(U[j] * Tn[j] for j in J_NORMAL)
    TP = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="TP")
    m.addConstr(TP == revenue - mfgcost - tard)

    # 3) 每个 α 只建 f[α] 与 z[α]，但 f[α] 由同一 TP/Cmax 线性给出
    f = {}
    z = {}
    for a in alphas:
        tag = f"a{str(a).replace('.','_')}"
        f[a] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"f_{tag}")
        m.addConstr(f[a] == a * TP - (1 - a) * 1000*Cmax, name=f"f_def_{tag}")
        z[a] = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"z[{tag}]")
        m.addConstr(z[a] >= T_value - f[a], name=f"hinge_{tag}")

    # 4) 目标：最小化 Σ z[a] + 极小的“空隙”项（让条带尽量贴紧）
    m.setObjective(quicksum(z[a] for a in alphas) + EPS_IDLE * quicksum(idle_terms), GRB.MINIMIZE)
    m.Params.MIPGap = 1e-3
    m.Params.TimeLimit = 1200
    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS(); m.write("phase2_iis.ilp")
        raise RuntimeError("Phase-2 infeasible")

    sol = {
        "PT_sum": sum(z[a].X for a in alphas),
        "z_by_alpha": {a: z[a].X for a in alphas},
        "f_by_alpha": {a: f[a].X for a in alphas},  # 现在会随 α 线性变化
        "TP": TP.X,
        "Cmax": Cmax.X,
        "s_b": {(p,j): s_b[(p,j)].X for p in range(3) for j in JOBS},
        "c_b": {(p,j): c_b[(p,j)].X for p in range(3) for j in JOBS},
        "s_e": {(q,j): s_e[(q,j)].X for q in range(5) for j in JOBS},
        "c_e": {(q,j): c_e[(q,j)].X for q in range(5) for j in JOBS},
        "s_p": {j: s_p[j].X for j in JOBS},
        "c_p": {j: c_p[j].X for j in JOBS},
        "order_pack": sorted(JOBS, key=lambda jj: s_p[jj].X)
    }
    return sol

# ---------- Gantt (every 5, makespan) ----------
COLOR = {
    "D1": "#009E73","D2": "#3C6DB4","D3": "#48C6EB",
    "N1": "#81C998","N2": "#F0A12C","N3": "#EF5E21","N4": "#B02226",
}
BODY_NAMES = ["Base","Big Arm","Wrist"]
ECAB_NAMES = ["Axis Computer Board","Axis Servo Drive","SMB Measurement Board","I/O Board","Main Computer Control"]
PACK_NAME = "Pipeline Pack assembly"

def plot_gantt(solution, legend_cols=4, dpi=150):
    rcParams["font.family"] = "Times New Roman"
    rcParams["font.size"] = 12
    rcParams["axes.titleweight"] = "bold"

    seg_b, seg_e = [], []
    for p in range(3):
        for j in JOBS:
            seg_b.append((BODY_NAMES[p], j, solution["s_b"][(p,j)], solution["c_b"][(p,j)]))
    for q in range(5):
        for j in JOBS:
            seg_e.append((ECAB_NAMES[q], j, solution["s_e"][(q,j)], solution["c_e"][(q,j)]))
    seg_p = [(PACK_NAME, j, solution["s_p"][j], solution["c_p"][j]) for j in solution["order_pack"]]

    fig = plt.figure(figsize=(14, 9), dpi=dpi, constrained_layout=True)
    gs  = gridspec.GridSpec(4,1,height_ratios=[1.2,3,6,2],figure=fig)
    ax_hdr = fig.add_subplot(gs[0,0]); ax_hdr.axis("off")
    ax_b   = fig.add_subplot(gs[1,0])
    ax_e   = fig.add_subplot(gs[2,0], sharex=ax_b)
    ax_p   = fig.add_subplot(gs[3,0], sharex=ax_b)

    def draw(ax, seg, title):
        lanes = {}
        for lane,_,_,_ in seg: lanes.setdefault(lane, len(lanes))
        for lane, j, s, f in seg:
            y = lanes[lane]
            ax.barh(y, f-s, left=s, height=0.56, color=COLOR[j], edgecolor="black", linewidth=0.8)
            ax.text((s+f)/2, y, j, ha="center", va="center")
        ax.set_yticks(list(lanes.values())); ax.set_yticklabels(list(lanes.keys()))
        ax.set_title(title); ax.set_xlabel("Time"); ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        ax.set_ylim(-0.8, len(lanes)-0.2)

    # 原三行替换为如下 6 行
    draw(ax_b, seg_b, "")  # 不在上方放标题
    ax_b.annotate("Gantt — Body assembly group", xy=(0.5, -0.35),
                  xycoords="axes fraction", ha="center", va="top",
                  fontsize=14, fontweight="bold", clip_on=False)

    draw(ax_e, seg_e, "")
    ax_e.annotate("Gantt — Electric cabinet assembly group", xy=(0.5, -0.18),
                  xycoords="axes fraction", ha="center", va="top",
                  fontsize=14, fontweight="bold", clip_on=False)

    draw(ax_p, seg_p, "")
    ax_p.annotate("Gantt — Pipeline pack assembly group", xy=(0.5, -0.50),
                  xycoords="axes fraction", ha="center", va="top",
                  fontsize=14, fontweight="bold", clip_on=False)

    # legend
    handles = [Patch(facecolor=COLOR[j], edgecolor="black", label=j) for j in JOBS]
    ax_hdr.text(0.0, 1.2, "Color → Order mapping", ha="left", va="top")
    leg = fig.legend(handles=handles, ncol=legend_cols, loc="upper left",
                     bbox_to_anchor=(0.01, 0.97), frameon=True, fancybox=True, borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.8); leg.get_frame().set_edgecolor("black")

    # every 5, makespan
    Cmax = max(solution["c_p"].values())
    xmax = int(math.ceil(Cmax/5.0))*5
    for ax in (ax_b, ax_e, ax_p):
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.35)
        ax.axvline(Cmax, color="black", linestyle="--", linewidth=1)
    ax_p.annotate(f"Makespan = {Cmax:.1f}", xy=(Cmax,0.0), xytext=(Cmax+0.8,0.8),
                  ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="->", lw=1),
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6))
    # 在 plt.show() 之前增加底部留白（防止下方标题被裁切）
    plt.subplots_adjust(bottom=0.15)
    plt.show()

# ---------- main ----------
if __name__ == "__main__":
    base = compute_T_alpha05()
    print(f"T(α=0.5)={base['T']:.3f}, TP={base['TP']:.3f}, Cmax={base['Cmax']:.3f}")

    robust = solve_min_sum_PT(T_value=base["T"], alphas=ALPHAS)
    print(f"\nMin Σ PT = {robust['PT_sum']:.3f}")
    for a in ALPHAS:
        print(f"  α={a:.1f}: z={robust['z_by_alpha'][a]:.3f}, f={robust['f_by_alpha'][a]:.3f}")
    print(f"\nPACK order: {robust['order_pack']}")

    plot_gantt(robust)
