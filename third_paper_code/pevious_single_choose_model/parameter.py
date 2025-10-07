import numpy as np
from typing import Dict, List, Tuple

# =========================
# 1) 基础参数与随机数据
# =========================
SEED = 12
np.random.seed(SEED)

num_special_orders = 100   # 特殊订单数量
num_normal_orders  = 25   # 普通订单数量

num_ops_body    = 3       # 本体装配工序数（单机、顺序加工）
num_ops_cabinet = 5       # 电柜装配工序数（单机、顺序加工）

# ---- 特殊订单（dominant jobs）----
sp_pt_body    = np.random.randint(1, 6, (num_ops_body,    num_special_orders))
sp_pt_cabinet = np.random.randint(1, 6, (num_ops_cabinet, num_special_orders))
sp_pt_pipe    = np.random.randint(1, 6, num_special_orders)
sp_pt_AC      = np.random.randint(300, 450, num_special_orders)

sp_release = np.random.randint(0, 50*(num_special_orders/10),  num_special_orders)               # r_j
sp_due     = sp_release + np.random.randint(30, 40, num_special_orders)  # d_j  (r<d)
sp_profit  = np.random.randint(30000, 50000, num_special_orders)

# ---- 普通订单（mandatory）----
nm_pt_body    = np.random.randint(1, 6, (num_ops_body,    num_normal_orders))
nm_pt_cabinet = np.random.randint(1, 6, (num_ops_cabinet, num_normal_orders))
nm_pt_pipe    = np.random.randint(1, 6, num_normal_orders)
nm_pt_AC      = np.random.randint(250, 350, num_normal_orders)

nm_due        = np.random.randint(20, 40, num_normal_orders)
nm_profit     = np.random.randint(10000, 20000, num_normal_orders)


# =========================
# 2) 预处理：折算单机工时 & 初筛
# =========================
sp_p_tilde = np.maximum(sp_pt_body.sum(axis=0), sp_pt_cabinet.sum(axis=0)) + sp_pt_pipe.astype(float)

# 初筛：考虑释放期窗口（更合理）
valid_mask = sp_p_tilde <= (sp_due - sp_release)

# 按有效掩码筛选特殊订单
sp_idx_kept = np.where(valid_mask)[0].tolist()

sp_pt_body    = sp_pt_body[:, sp_idx_kept]
sp_pt_cabinet = sp_pt_cabinet[:, sp_idx_kept]
sp_pt_pipe    = sp_pt_pipe[sp_idx_kept]
sp_release    = sp_release[sp_idx_kept]
sp_due        = sp_due[sp_idx_kept]
sp_pt_AC = sp_pt_AC[sp_idx_kept]
sp_profit     = sp_profit[sp_idx_kept]
sp_p_tilde    = sp_p_tilde[sp_idx_kept]

num_special_orders = sp_pt_body.shape[1]  # 更新数量
print("特殊订单数量",num_special_orders)

# =========================
# 3) 构造统一订单列表（字典），区分两类
#    —— 适配“先后顺序”决策变量
# =========================
# 统一的全局订单 ID：先普通后特殊（也可以反过来，只要保持一致映射）
normal_ids  = list(range(1, num_normal_orders + 1))
special_ids = list(range(num_normal_orders + 1, num_normal_orders + num_special_orders + 1))

orders: List[Dict] = []

# 普通订单（必须生产；无 r/d）
for j, oid in enumerate(normal_ids):
    orders.append({
        "id": oid,
        "type": "normal",
        "is_special": False,
        "proc_body":    nm_pt_body[:, j].astype(float),    # shape=(3,)
        "proc_cabinet": nm_pt_cabinet[:, j].astype(float), # shape=(5,)
        "proc_pipe":    float(nm_pt_pipe[j]),
        "release": 0.0,
        "due": float(nm_due[j]),
        "profit": float(nm_profit[j]),
        "AC": float(nm_pt_AC[j]),
        # 选择阶段不涉及普通单，这里可不计算 p_tilde；保留占位方便后续统一接口
        "p_tilde": None
    })

# 特殊订单（可选；有 r/d）
for j, oid in enumerate(special_ids):
    orders.append({
        "id": oid,
        "type": "special",
        "is_special": True,
        "proc_body":    sp_pt_body[:, j].astype(float),
        "proc_cabinet": sp_pt_cabinet[:, j].astype(float),
        "proc_pipe":    float(sp_pt_pipe[j]),
        "release": float(sp_release[j]),
        "due":     float(sp_due[j]),
        "profit":  float(sp_profit[j]),
        "AC": float(sp_pt_AC[j]),
        "p_tilde": float(sp_p_tilde[j])  # 给选择阶段用
    })

# 全局索引映射（方便建 z_{i,k} 的上三角索引集）
id2idx = {o["id"]: i for i, o in enumerate(orders)}   # 0..N-1
idx2id = {i: o["id"] for i, o in enumerate(orders)}
N = len(orders)

# =========================
# 4) 生成“先后关系”索引集（适用于 z_{i,k}）
#    —— 适配你的顺序变量建模：每条产线一套
# =========================
# 这里 i<k 即可（单向变量更简洁）：z^ell_{i,k}=1 表示 i 在 k 之前
pairs_all: List[Tuple[int, int]] = [(i, k) for i in range(N) for k in range(i+1, N)]

index_sets = {
    "pairs_body":    pairs_all.copy(),  # 本体线用
    "pairs_cabinet": pairs_all.copy(),  # 电柜线用
    "pairs_pipe":    pairs_all.copy(),  # 装配线用
}

# =========================
# 5) Big-M（紧上界）按资源分别给出，利于数值稳定
# =========================
def tight_M_for_line(orders_list: List[Dict], which: str) -> float:
    """
    which: 'body' / 'cabinet' / 'pipe'
    """
    if which == "body":
        return float(sum(o["proc_body"].sum() for o in orders_list))
    if which == "cabinet":
        return float(sum(o["proc_cabinet"].sum() for o in orders_list))
    if which == "pipe":
        return float(sum(o["proc_pipe"] for o in orders_list))
    raise ValueError("unknown line")

Mb = tight_M_for_line(orders, "body")
Me = tight_M_for_line(orders, "cabinet")
Mp = tight_M_for_line(orders, "pipe")

big_M = {"body": Mb, "cabinet": Me, "pipe": Mp}

# =========================
# 6) 选择阶段数据打包（仅特殊订单）
# =========================
special_mask = [o["is_special"] for o in orders]
special_list = [o for o in orders if o["is_special"]]

selection_ds = {
    "ids":       [o["id"] for o in special_list],
    "release":   [o["release"] for o in special_list],
    "due":       [o["due"] for o in special_list],
    "p_tilde":   [o["p_tilde"] for o in special_list],
    "profit":    [o["profit"] for o in special_list],
    "AC":        [o["AC"] for o in special_list],
    # 可直接构造净利润：profit - AC * p_tilde
    "net_profit":[o["profit"] - o["AC"] * o["p_tilde"] for o in special_list],
}

# =========================
# 7) 打印关键信息
# =========================
print(f"共生成订单 N={N}，其中普通 {len(normal_ids)}，特殊(初筛后) {len(special_ids)}")
print("紧的 Big-M:", big_M)
print("示例-选择阶段数据(前3条)：")
for i in range(min(3, len(selection_ds['ids']))):
    print(f"  ID={selection_ds['ids'][i]}, r={selection_ds['release'][i]}, d={selection_ds['due'][i]}, "
          f"p~={selection_ds['p_tilde'][i]:.1f}, net={selection_ds['net_profit'][i]:.1f}")
