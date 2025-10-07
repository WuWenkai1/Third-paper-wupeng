# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec, rcParams
from matplotlib.ticker import MultipleLocator
import math


# ---------------- 基础数据 ----------------
J_SPECIAL = ["D1","D2","D3"]
J_NORMAL  = ["N1","N2","N3","N4"]
JOBS = J_SPECIAL + J_NORMAL

# BODY（3 串行工位）
P_BODY = {
    "D1":[5,3,5], "D2":[3,5,3], "D3":[2,3,2],
    "N1":[5,2,4], "N2":[4,3,4], "N3":[2,5,3], "N4":[5,4,2],
}
# ECAB（5 串行工位）
P_ECAB = {
    "D1":[1,3,2,4,5], "D2":[2,3,2,3,2], "D3":[1,3,2,1,3],
    "N1":[2,4,2,4,1], "N2":[5,1,2,3,1], "N3":[2,4,2,1,4], "N4":[4,3,1,1,3],
}
# PACK（单机）
P_PACK = {"D1":3,"D2":4,"D3":5,"N1":3,"N2":2,"N3":3,"N4":2}

# 释放期 / 交期
R   = {"D1":5,"D2":20,"D3":25,"N1":0,"N2":0,"N3":0,"N4":0}
DUE = {"D1":25,"D2":40,"D3":43,"N1":25,"N2":20,"N3":15,"N4":25}

# 经济参数
PI  = {"D1":39000,"D2":30000,"D3":27000,"N1":13500,"N2":12400,"N3":13300,"N4":15900}
U   = {"N1":20,"N2":20,"N3":25,"N4":20}
AC  = {"D1":500,"D2":442,"D3":403,"N1":318,"N2":275,"N3":294,"N4":272}

# 机器名（甘特图）
BODY_NAMES = ["Base","Big Arm","Wrist"]
ECAB_NAMES = ["Main Computer Control","Axis Computer Board","Axis Servo Drive",
              "SMB Measurement Board","I/O Board"]
PACK_NAME = "Pipeline Pack assembly"

# 调色
COLOR = {
    "D1": "#009E73", "D2": "#3C6DB4", "D3": "#48C6EB",
    "N1": "#81C998", "N2": "#F0A12C", "N3": "#EF5E21", "N4": "#B02226",
}

# ---------- 预约构造：以“释放即开工、各线无等待”为目标（仅作上界参考） ----------
def reservation_for_special(j):
    tb = [0]*3; te = [0]*5
    tb[0] = R[j]
    for p in range(1,3): tb[p] = tb[p-1] + P_BODY[j][p-1]
    tb_end = tb[2] + P_BODY[j][2]
    te[0] = R[j]
    for q in range(1,5): te[q] = te[q-1] + P_ECAB[j][q-1]
    te_end = te[4] + P_ECAB[j][4]
    t_pack0 = max(tb_end, te_end)              # 理想的PACK起点
    res = {('B',p):(tb[p], tb[p]+P_BODY[j][p]) for p in range(3)}
    res.update({('E',q):(te[q], te[q]+P_ECAB[j][q]) for q in range(5)})
    res[('P',0)] = (t_pack0, t_pack0 + P_PACK[j])
    return res

def can_place(start, dur, res_seg):
    if res_seg is None: return True
    t0, _ = res_seg
    return start + dur <= t0 + 1e-9

def normal_key(j):  # EDD → FCFS
    return (DUE[j], R[j])

# ---------- 主调度（绝对优先 + 预测占用 + 修正PACK开始式） ----------
def schedule_strict():
    # 机器可用时间
    avail_b = [0.0]*3
    avail_e = [0.0]*5
    avail_p = 0.0

    # 时标记录
    s_b, c_b, s_e, c_e, s_p, c_p = {}, {}, {}, {}, {}, {}

    # 各订单下一道工序索引
    b_idx = {j:0 for j in JOBS}
    e_idx = {j:0 for j in JOBS}

    # 特殊订单按释放排序
    specials = sorted(J_SPECIAL, key=lambda j: R[j])

    # —— 逐个特殊订单处理 —— #
    for s in specials:
        res = reservation_for_special(s)

        # 1) 在该特殊预约前推进普通，但必须保证“不会与预约重叠”
        progressed = True
        while progressed:
            progressed = False
            cand = []

            # BODY 可开的普通工序
            for j in J_NORMAL:
                p = b_idx[j]
                if p < 3:
                    pred = 0.0 if p==0 else c_b[(p-1,j)]
                    est = max(R[j] if p==0 else pred, avail_b[p])
                    if can_place(est, P_BODY[j][p], res.get(('B',p))):
                        cand.append(('B', est, j))

            # ECAB 可开的普通工序
            for j in J_NORMAL:
                q = e_idx[j]
                if q < 5:
                    pred = 0.0 if q==0 else c_e[(q-1,j)]
                    est = max(R[j] if q==0 else pred, avail_e[q])
                    if can_place(est, P_ECAB[j][q], res.get(('E',q))):
                        cand.append(('E', est, j))

            # PACK：两线完成且不与预约重叠
            for j in J_NORMAL:
                if b_idx[j]==3 and e_idx[j]==5 and (j not in s_p):
                    est = max(avail_p, c_b[(2,j)], c_e[(4,j)])
                    if can_place(est, P_PACK[j], res.get(('P',0))):
                        cand.append(('P', est, j))

            if not cand: break
            cand.sort(key=lambda x: (x[1], normal_key(x[2])))
            line, start, j = cand[0]

            if line == 'B':
                p = b_idx[j]
                s_b[(p,j)] = start
                c_b[(p,j)] = start + P_BODY[j][p]
                avail_b[p]  = c_b[(p,j)]
                b_idx[j] += 1
                progressed = True

            elif line == 'E':
                q = e_idx[j]
                s_e[(q,j)] = start
                c_e[(q,j)] = start + P_ECAB[j][q]
                avail_e[q]  = c_e[(q,j)]
                e_idx[j] += 1
                progressed = True

            else:  # 'P'
                s_p[j] = start
                c_p[j] = start + P_PACK[j]
                avail_p  = c_p[j]
                progressed = True

        # 2) 到释放时刻：插入特殊订单（关键：开始时刻用 max(...)，而不是预约 t0）
        # BODY
        for p in range(3):
            prev = c_b[(p-1,s)] if p>0 else 0.0
            start = max(R[s] if p==0 else prev, avail_b[p])
            s_b[(p,s)] = start
            c_b[(p,s)] = start + P_BODY[s][p]
            avail_b[p]  = c_b[(p,s)]
        # ECAB
        for q in range(5):
            prev = c_e[(q-1,s)] if q>0 else 0.0
            start = max(R[s] if q==0 else prev, avail_e[q])
            s_e[(q,s)] = start
            c_e[(q,s)] = start + P_ECAB[s][q]
            avail_e[q]  = c_e[(q,s)]
        # PACK（修正点）
        start_pack = max(avail_p, c_b[(2,s)], c_e[(4,s)])
        s_p[s] = start_pack
        c_p[s] = start_pack + P_PACK[s]
        avail_p  = c_p[s]

    # 3) 全部特殊完成后，把剩余普通排完（无预约限制）
    still = True
    while still:
        still = False
        cand = []
        for j in J_NORMAL:
            p = b_idx[j]
            if p < 3:
                pred = 0.0 if p==0 else c_b[(p-1,j)]
                est = max(R[j] if p==0 else pred, avail_b[p])
                cand.append(('B', est, j))
        for j in J_NORMAL:
            q = e_idx[j]
            if q < 5:
                pred = 0.0 if q==0 else c_e[(q-1,j)]
                est = max(R[j] if q==0 else pred, avail_e[q])
                cand.append(('E', est, j))
        for j in J_NORMAL:
            if b_idx[j]==3 and e_idx[j]==5 and (j not in s_p):
                est = max(avail_p, c_b[(2,j)], c_e[(4,j)])
                cand.append(('P', est, j))

        if not cand: break
        cand.sort(key=lambda x: (x[1], normal_key(x[2])))
        line, start, j = cand[0]

        if line == 'B':
            p = b_idx[j]
            s_b[(p,j)] = start
            c_b[(p,j)] = start + P_BODY[j][p]
            avail_b[p]  = c_b[(p,j)]
            b_idx[j] += 1
            still = True

        elif line == 'E':
            q = e_idx[j]
            s_e[(q,j)] = start
            c_e[(q,j)] = start + P_ECAB[j][q]
            avail_e[q]  = c_e[(q,j)]
            e_idx[j] += 1
            still = True

        else:
            s_p[j] = start
            c_p[j] = start + P_PACK[j]
            avail_p  = c_p[j]
            still = True

    # 安全性：所有订单必须进入 PACK
    for j in JOBS:
        assert j in s_p and j in c_p, f"{j} 未进入PACK"

    order_pack = sorted(JOBS, key=lambda jj: s_p[jj])
    return {"s_b":s_b,"c_b":c_b,"s_e":s_e,"c_e":c_e,"s_p":s_p,"c_p":c_p,"order_pack":order_pack}

# ---------- 经济评价 ----------
def evaluate_PT(sol, alphas):
    revenue  = sum(PI[j] for j in JOBS)
    mfg_cost = sum(AC[j]*(sum(P_BODY[j])+sum(P_ECAB[j])+P_PACK[j]) for j in JOBS)
    tard = sum(U[j]*max(0.0, sol["c_p"][j]-DUE[j]) for j in J_NORMAL)
    TP = revenue - mfg_cost - tard
    Cmax = max(sol["c_p"].values())
    T = 21865
    PT =[]
    rows = []
    for a in alphas:
        f = a*TP - (1-a)*1000*Cmax
        S = max(0.0, T-f)
        PT.append(S)
        rows.append({"alpha":a, "f":f, "PT":S})
    return TP, Cmax, T, PT, rows

# ---------- 甘特图 ----------
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

# ---------- 主程 ----------
if __name__ == "__main__":
    sol = schedule_strict()
    # 关键检查：D2 的 PACK 必须 32–36
    print(f"D2 PACK start={sol['s_p']['D2']:.1f}, finish={sol['c_p']['D2']:.1f} (should be 32–36)")
    plot_gantt(sol)
    TP, Cmax, T, PT, rows = evaluate_PT(sol,alphas=[0.3,0.4,0.5,0.6,0.7])
    print("PT：",PT)
    Sum_PT = sum(PT)
    print(f"\nTP={TP:.2f}, Cmax={Cmax:.2f}, T(α=0.5)={T:.2f}")
    print("PT(X) by α:")
    print("Final_PT:",Sum_PT)
    for r in rows:
        print(f"  α={r['alpha']:.1f}  f={r['f']:.2f}  PT={r['PT']:.2f}")
