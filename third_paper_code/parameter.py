import numpy as np
from typing import Dict, List, Tuple

# =========================
# 1) 基础参数与随机数据
# =========================
SEED = 12
np.random.seed(SEED)

num_special_orders = 100  # 特殊订单数量
num_normal_orders  = 100  # 普通订单数量

num_special_orders_copy = num_special_orders
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
sp_delay_penalty = np.random.randint(1e6, 1e9, num_special_orders)

# ---- 普通订单（mandatory）----
nm_pt_body    = np.random.randint(1, 6, (num_ops_body,    num_normal_orders))
nm_pt_cabinet = np.random.randint(1, 6, (num_ops_cabinet, num_normal_orders))
nm_pt_pipe    = np.random.randint(1, 6, num_normal_orders)
nm_pt_AC      = np.random.randint(250, 350, num_normal_orders)

nm_due        = np.random.randint(20, 300, num_normal_orders)
nm_profit     = np.random.randint(10000, 20000, num_normal_orders)
nm_delay_penalty = np.random.randint(40, 100, num_normal_orders)


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
        "penalty": float(nm_delay_penalty[j]),
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
        "penalty": float(sp_delay_penalty[j]),
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
#print("特殊订单信息：",special_list)
selection_ds = {
    "ids":       [o["id"] for o in special_list],
    "type": "special",
    "is_special": True,
    "release":   [o["release"] for o in special_list],
    "due":       [o["due"] for o in special_list],
    "p_tilde":   [o["p_tilde"] for o in special_list],
    "profit":    [o["profit"] for o in special_list],
    "AC":        [o["AC"] for o in special_list],
    "penalty": [o["penalty"] for o in special_list],
    # 可直接构造净利润：profit - AC * p_tilde
    "net_profit":[o["profit"] - o["AC"] * o["p_tilde"] for o in special_list],
}
# =========================
# 8) 选择“特殊订单子集” + 生成两个字典 + 生成可用顺序
# =========================
# 说明：
# - n25/n50/n75/n100：在 special_list 中的 1-based 位置索引（不是实际订单ID）
# - special_25/special_50/...：真实的特殊订单 ID 顺序，用于检测（想按此顺序去跑）
# 目标：
# - selected_specials_dict: 仅选中的特殊订单（按真实ID做 key）
# - combined_orders_dict:   选中特殊订单 + 所有普通订单（按真实ID做 key）
# - special_seq_indices:    把 special_X 的“真实ID顺序”映射为 orders_list（=special_orders）的“局部索引顺序”，可直接喂给 simulate_three_perms


special_25 = [107, 114, 112, 118, 109, 108, 125, 123, 120, 106]  # （示例：若你需要25档的顺序，可补全）
seq_id_25 = [x - 100 + num_normal_orders for x in special_25]

special_50 = [119, 132, 137, 128, 146, 127, 123, 126, 101, 129, 140, 125, 124, 145, 114, 134, 147, 131]  # 真实ID顺序
seq_id_50 = [x - 100 + num_normal_orders for x in special_50]

special_75 = [104, 121, 106, 154, 155, 160, 119, 113, 149, 152, 124, 144, 151, 157, 118, 133, 114, 158, 128, 102, 129, 123, 103, 159, 145]  # 真实ID顺序
seq_id_75 = [x - 100 + num_normal_orders for x in special_75]

special_100 = [147, 150, 157, 118, 152, 133, 128, 124, 180, 119, 174, 145, 169, 141, 125, 178, 163, 176, 121, 173, 179, 126, 153, 171, 123, 109, 182, 160, 197, 106, 146, 199]  # 真实ID顺序, len = 32
seq_id_100 = [x - 100 + num_normal_orders for x in special_100]

# —— 工具函数：从 1-based 位置列表抽取特殊订单对象（返回 list）
def pick_specials_by_ids(id_list, special_list):

    id_set = set(id_list)
    picked = [o for o in special_list if o["id"] in id_set]
    missing = [oid for oid in id_list if oid not in {o["id"] for o in picked}]
    if missing:
        print(f"[warn] IDs not found in current filtered special_list: {missing}")
    return picked



# —— 工具函数：把“真实ID顺序”映射为“当前 orders_list（=special_orders）的局部索引顺序”
def _ids_to_local_indices(id_order, local_orders):
    id2local = {o["id"]: idx for idx, o in enumerate(local_orders)}
    seq = []
    for oid in id_order:
        if oid not in id2local:
            # 若某个真实ID不在当前选中的 special_orders 里，检测功能，按理说是不会打印的，这里选择跳过并告警,
            print(f"[warn] special ID {oid} not in the selected subset; skipped.")
            continue
        seq.append(id2local[oid])
    return seq

# —— 主分支：根据 num_special_orders_copy 选择对应的集合
order_list = []      # 供后续“全局顺序”逻辑使用的容器（如你需要）
normal_orders = []   # 普通订单列表（保留你的结构）
for o in orders:
    if o["type"] == "normal":
        order_list.append(o)
        normal_orders.append(o)

selected_specials = []   # list of special order dicts（被选中的特殊订单）
special_orders = []      # 等价于 selected_specials，用于 simulate_three_perms 的 orders_list
special_seq_indices = [] # 可直接喂给 simulate_three_perms 的“可用顺序”（局部索引序列）

# 选择对应规模
if num_special_orders_copy == 25:
    selected_specials = pick_specials_by_ids(seq_id_25, special_list)
    special_orders = selected_specials[:]  # 用于检测函数的 orders_list= special_orders
    special_seq_indices = _ids_to_local_indices(special_25, special_orders)

elif num_special_orders_copy == 50:
    selected_specials = pick_specials_by_ids(seq_id_50, special_list)
    special_orders = selected_specials[:]
    special_seq_indices = _ids_to_local_indices(special_50, special_orders)

elif num_special_orders_copy == 75:
    selected_specials = pick_specials_by_ids(seq_id_75, special_list)
    special_orders = selected_specials[:]
    special_seq_indices = _ids_to_local_indices(special_75, special_orders)

elif num_special_orders_copy == 100:
    selected_specials = pick_specials_by_ids(seq_id_100, special_list)
    special_orders = selected_specials[:]
    special_seq_indices = _ids_to_local_indices(special_100, special_orders)

# —— 1) 选中特殊订单字典（key=真实ID）
selected_specials_dict = {o["id"]: o for o in selected_specials}

# —— 2) 合并字典 = 选中特殊订单 + 全部普通订单（key=真实ID）
combined_orders_dict = {}
combined_orders_dict.update({o["id"]: o for o in normal_orders})
combined_orders_dict.update(selected_specials_dict)

print("最终字典：", combined_orders_dict)
# ——（可选）打印检查
print("\n[check] selected_specials_dict keys (IDs):", sorted(selected_specials_dict.keys())[:], "...")
print("[check] combined_orders_dict size:", len(combined_orders_dict))

# =========================
# 9) 检测：simulate_three_perms
# =========================
def simulate_three_perms(
    order_number: int,
    perm_body: List[int],
    perm_cab: List[int],
    perm_pipe: List[int],
    orders_list: List[Dict],
):
    n = order_number

    # ===== 本体线（3 串行工位）=====
    m1 = 3
    S_body = np.zeros((n, m1))
    C_body = np.zeros((n, m1))
    idx_body = {job: j_idx for j_idx, job in enumerate(perm_body)}
    for j_idx, job in enumerate(perm_body):
        proc = orders_list[job]["proc_body"]
        release = float(orders_list[job]["release"])
        for k in range(m1):
            if j_idx == 0 and k == 0:
                start = max(0.0, release)
            elif j_idx == 0:
                start = C_body[j_idx, k - 1]
            elif k == 0:
                start = max(C_body[j_idx - 1, k], release)
            else:
                start = max(C_body[j_idx, k - 1], C_body[j_idx - 1, k])
            S_body[j_idx, k] = start
            C_body[j_idx, k] = start + float(proc[k])
    body_finish = {perm_body[j]: float(C_body[j, -1]) for j in range(len(perm_body))}

    # ===== 电柜线（5 串行工位）=====
    m2 = 5
    S_cab = np.zeros((n, m2))
    C_cab = np.zeros((n, m2))
    idx_cab = {job: j_idx for j_idx, job in enumerate(perm_cab)}
    for j_idx, job in enumerate(perm_cab):
        proc = orders_list[job]["proc_cabinet"]
        release = float(orders_list[job]["release"])
        for k in range(m2):
            if j_idx == 0 and k == 0:
                start = max(0.0, release)
            elif j_idx == 0:
                start = C_cab[j_idx, k - 1]
            elif k == 0:
                start = max(C_cab[j_idx - 1, k], release)
            else:
                start = max(C_cab[j_idx, k - 1], C_cab[j_idx - 1, k])
            S_cab[j_idx, k] = start
            C_cab[j_idx, k] = start + float(proc[k])
    cab_finish = {perm_cab[j]: float(C_cab[j, -1]) for j in range(len(perm_cab))}

    # ===== 管线包（单机）=====
    pipe_finish: Dict[int, float] = {}
    prev_finish = 0.0
    tardy_count = 0

    for job in perm_pipe:
        # 三线汇合后的开工时间（管线包的开始）
        ready = max(prev_finish, body_finish[job], cab_finish[job])
        pf = ready + float(orders_list[job]["proc_pipe"])
        pipe_finish[job] = pf
        prev_finish = pf

        # 取各线“开始加工时间”（首工位的 S）
        # 注意：job 是 orders_list 的局部索引，要通过 idx_* 找到对应行
        b_row = idx_body[job]
        c_row = idx_cab[job]
        start_body_first = float(S_body[b_row, 0])
        start_cab_first  = float(S_cab[c_row, 0])
        start_pipe       = float(ready)

        # 其他信息
        oid   = orders_list[job]["id"]
        Re    = float(orders_list[job]["release"])
        pro_b = orders_list[job]["proc_body"]
        pro_c = orders_list[job]["proc_cabinet"]
        pro_p = float(orders_list[job]["proc_pipe"])
        due   = float(orders_list[job]["due"])
        late  = max(0.0, pf - due)
        if late > 0.0:
            tardy_count += 1

        print(
            f"特殊订单 {oid} | "
            f"本体开始: {start_body_first:.1f}, 电柜开始: {start_cab_first:.1f}, 管线包开始: {start_pipe:.1f} | "
            f"完工: {pf:.1f}, 释放: {Re:.1f}, 交期: {due:.1f}, 延期: {late:.1f} | "
            f"本体工时: {pro_b}, 电柜工时: {pro_c}, 管线包工时: {pro_p}"
        )

    return tardy_count

print("\n[check] usable sequence indices for detection:", special_seq_indices)

# 按同一顺序分别喂 body/cab/pipe（你原检测函数三条产线用同一顺序）
tardy_count = simulate_three_perms(
    order_number=len(special_orders),
    perm_body=special_seq_indices,
    perm_cab=special_seq_indices,
    perm_pipe=special_seq_indices,
    orders_list=special_orders
)
print("延迟次数：", tardy_count)
