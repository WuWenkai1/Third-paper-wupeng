import numpy as np

# 定义常量和参数
num_special_orders = 30  # 特殊订单数量
num_normal_orders = 30   # 普通订单数量
num_machines_body = 3    # 本体组装机器数量
num_machines_cabinet = 5 # 电柜组装机器数量

# 随机生成参数
np.random.seed(42)

# 特殊订单参数
special_processing_times_body = np.random.randint(1, 6, (num_machines_body, num_special_orders))
special_processing_times_cabinet = np.random.randint(1, 6, (num_machines_cabinet, num_special_orders))
special_processing_times_pipeline = np.random.randint(1, 6, num_special_orders)
special_release_times = np.random.randint(0, 20, num_special_orders)
special_due_times = np.random.randint(10, 30, num_special_orders)
special_profits = np.random.randint(3000, 5000, num_special_orders)

# 普通订单参数
normal_processing_times_body = np.random.randint(1, 6, (num_machines_body, num_normal_orders))
normal_processing_times_cabinet = np.random.randint(1, 6, (num_machines_cabinet, num_normal_orders))
normal_processing_times_pipeline = np.random.randint(1, 6, num_normal_orders)
normal_profits = np.random.randint(1000, 3000, num_normal_orders)

AC1 = 50
AC2 = 100

# 将所有参数存储在字典中
orders_parameters = {
    "special": {
        "num_orders": num_special_orders,
        "processing_times_body": special_processing_times_body,
        "processing_times_cabinet": special_processing_times_cabinet,
        "processing_times_pipeline": special_processing_times_pipeline,
        "release_times": special_release_times,
        "due_times": special_due_times,
        "profits": special_profits,
        "order_ids": list(range(1, num_special_orders + 1))  # 为每个特殊订单编上序号
    },
    "normal": {
        "num_orders": num_normal_orders,
        "processing_times_body": normal_processing_times_body,
        "processing_times_cabinet": normal_processing_times_cabinet,
        "processing_times_pipeline": normal_processing_times_pipeline,
        "profits": normal_profits,
        "order_ids": list(range(num_special_orders + 1, num_special_orders + num_normal_orders + 1))  # 为每个普通订单编上序号
    },
    "AC1": AC1,
    "AC2": AC2
}


# 提取特殊订单和普通订单的参数
special_orders = orders_parameters["special"]
normal_orders = orders_parameters["normal"]
AC1 = orders_parameters["AC1"]
AC2 = orders_parameters["AC2"]

M = 1e6  # 大数，用于约束条件

# 单个时间求解
total_processing_time_single = np.zeros(special_orders["num_orders"])  # 初始化特殊订单总加工时长矩阵

for j in range(special_orders["num_orders"]):
    total_processing_time_body = np.sum(special_orders["processing_times_body"][:, j])
    total_processing_time_cabinet = np.sum(special_orders["processing_times_cabinet"][:, j])
    total_processing_time_single[j] = max(total_processing_time_body, total_processing_time_cabinet) + \
                                      special_orders["processing_times_pipeline"][j]

# 初步筛选条件
valid_indices = [i for i in range(special_orders["num_orders"]) if total_processing_time_single[i] <= special_orders["due_times"][i]]

# 根据有效索引筛选订单
special_orders["processing_times_body"] = special_orders["processing_times_body"][:, valid_indices]
special_orders["processing_times_cabinet"] = special_orders["processing_times_cabinet"][:, valid_indices]
special_orders["processing_times_pipeline"] = special_orders["processing_times_pipeline"][valid_indices]
special_orders["release_times"] = special_orders["release_times"][valid_indices]
special_orders["due_times"] = special_orders["due_times"][valid_indices]
special_orders["profits"] = special_orders["profits"][valid_indices]
total_processing_time_single = total_processing_time_single[valid_indices]

special_orders["num_orders"] = len(valid_indices)  # 更新特殊订单数量
special_orders["order_ids"] = [special_orders["order_ids"][i] for i in valid_indices] #更新特殊订单编号

print("选出的订单：", special_orders["order_ids"])

#总计数量
num_special_orders = special_orders["num_orders"]
num_normal_orders = normal_orders["num_orders"]
num_orders = num_special_orders + num_normal_orders

num_positions = num_orders

# 初始化决策变量矩阵
X = np.zeros((num_positions, num_orders), dtype=int) #本体顺序
Y = np.zeros((num_positions, num_orders), dtype=int) #电柜顺序
Z = np.zeros((num_positions, num_orders), dtype=int) #管线包顺序
y = np.zeros(num_special_orders, dtype=int)          #特殊订单是否被选择，y[i] = 1 表示特殊订单 i 被选择；否则为 0

# 定义适应度函数
def fitness_function(schedule):

    # 初始化每台机器的开始时间和完成时间
    start_times_body = np.zeros((num_machines_body, num_orders))
    completion_times_body = np.zeros((num_machines_body, num_orders))
    start_times_cabinet = np.zeros((num_machines_cabinet, num_orders))
    completion_times_cabinet = np.zeros((num_machines_cabinet, num_orders))
    start_times_pipeline = np.zeros(num_orders)
    completion_times_pipeline = np.zeros(num_orders)

    remember = []

    for order_index in range(num_orders):
        if order_index < num_special_orders:
            # 特殊订单
            processing_times_body = special_orders["processing_times_body"]
            processing_times_cabinet = special_orders["processing_times_cabinet"]
            processing_times_pipeline = special_orders["processing_times_pipeline"]
            release_time = special_orders["release_times"][order_index]
            due_time = special_orders["due_times"][order_index]
        else:
            # 普通订单
            order_index_normal = order_index - num_special_orders
            processing_times_body = normal_orders["processing_times_body"]
            processing_times_cabinet = normal_orders["processing_times_cabinet"]
            processing_times_pipeline = normal_orders["processing_times_pipeline"]
            release_time = 0
            due_time = np.inf

    for X, Y, Z, y in schedule:

        # 计算本体组装的开始时间和完成时间
        for p in range(num_machines_body):
            for j in range(num_orders):
                start_times_body[p, j] = max(0, start_times_body[p, j])  # 确保非负
                completion_times_body[p, j] = start_times_body[p, j] + processing_times_body[p, j]
                if p > 0:
                    start_times_body[p, j] = max(start_times_body[p, j], completion_times_body[p-1, j])

            # 位置的约束条件
            for u in range(num_orders):
                for v in range(num_orders):
                    if u != v:
                        for i in range(num_positions-1):
                            start_times_body[p, u] = max(start_times_body[p, u], completion_times_body[p, v] + M * (X[i+1][u] - 1) + M * (X[i][v] - 1))

        # 计算电柜组装的开始时间和完成时间
        for q in range(num_machines_cabinet):
            for j in range(num_orders):
                start_times_cabinet[q, j] = max(0, start_times_cabinet[q, j])  # 确保非负
                completion_times_cabinet[q, j] = start_times_cabinet[q, j] + processing_times_cabinet[q, j]
                if q > 0:
                    start_times_cabinet[q, j] = max(start_times_cabinet[q, j], completion_times_cabinet[q-1, j])

            # 位置的约束条件
            for u in range(num_orders):
                for v in range(num_orders):
                    if u != v:
                        for i in range(num_positions-1):
                            start_times_cabinet[q, u] = max(start_times_cabinet[q, u], completion_times_cabinet[q, v] + M * (Y[i+1][u] - 1) + M * (Y[i][v] - 1))

        # 计算管道包装的开始时间和完成时间
        for j in range(num_orders):
            arrival_time = max(completion_times_body[:, j].max(), completion_times_cabinet[:, j].max())
            start_times_pipeline[j] = max(start_times_pipeline[j], arrival_time)
            completion_times_pipeline[j] = start_times_pipeline[j] + processing_times_pipeline[j]

        # 位置的约束条件
        for u in range(num_orders):
            for v in range(num_orders):
                if u != v:
                    for i in range(num_positions - 1):
                        start_times_pipeline[u] = max(start_times_pipeline[u], completion_times_pipeline[v] + M * (Z[i+1][u] - 1) + M * (Z[i][v] - 1))

        # 对特殊订单的约束
        for p in range(num_machines_body):
            for q in range(num_machines_cabinet):
                for j in range(num_special_orders):
                    start_times_body[p, j] = max(start_times_body[p, j], special_orders["release_times"][j] * y[j])
                    start_times_cabinet[q, j] = max(start_times_cabinet[q, j], special_orders["release_times"][j] * y[j])
                    due_time = special_orders["due_times"][j]
                    completion_times_pipeline[j] = min(completion_times_pipeline[j], due_time)

        # 对决策变量的约束
        for i in range(num_positions):
            assert np.sum([X[i][j] for j in range(num_orders)]) == 1
            assert np.sum([Y[i][j] for j in range(num_orders)]) == 1
            assert np.sum([Z[i][j] for j in range(num_orders)]) == 1
            y[g] = np.sum([X[i][g] for g in range(num_special_orders)])

        for j in range(num_orders):
            assert np.sum([X[i][j] for i in range(num_positions)]) == 1
            assert np.sum([Y[i][j] for i in range(num_positions)]) == 1
            assert np.sum([Z[i][j] for i in range(num_positions)]) == 1

        remember.append(completion_times_pipeline[order_index])
        C_max = remember.max[]

    return completion_times_pipeline, remember, C_max


# 利润函数
def profit_cost_function(schedule, remember):

    for y in schedule:
        for i in range(num_special_orders):
            proportion_single = (total_processing_time_single[i] * y[i]) / (total_processing_time_single[i] * y[i].sum())
        sing_cost =


    total_cost = AC2 * (max(special_orders["release_times"][selected_orders[:num_special_orders]]) - min(special_orders["release_times"][selected_orders[:num_special_orders]]))
    single_profit = []
    for i in range(len(schedule)):
        order_index = schedule[i]
        if selected_orders[order_index] == 0:
            single_profit.append(0)
            continue

        if order_index < num_special_orders:
            single_cost = (total_processing_time_single[order_index] / np.sum(
                total_processing_time_single[selected_orders[:num_special_orders]])) * total_cost
            single_profit_1 = special_orders["profits"][order_index] - single_cost
        else:
            single_cost = (total_processing_time_single[order_index - num_special_orders] / np.sum(
                total_processing_time_single[selected_orders[num_special_orders:]])) * total_cost
            single_profit_1 = normal_orders["profits"][order_index - num_special_orders] - single_cost

        single_profit.append(single_profit_1)

    return single_profit, total_cost