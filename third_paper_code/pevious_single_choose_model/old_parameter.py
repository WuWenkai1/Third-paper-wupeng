import numpy as np

# 定义常量和参数
num_special_orders = 30  # 特殊订单数量
num_normal_orders = 30   # 普通订单数量
num_machines_body = 3    # 本体组装机器数量
num_machines_cabinet = 5 # 电柜组装机器数量

# 随机生成参数
np.random.seed(12)

# 特殊订单参数
special_processing_times_body = np.random.randint(1, 6, (num_machines_body, num_special_orders))
special_processing_times_cabinet = np.random.randint(1, 6, (num_machines_cabinet, num_special_orders))
special_processing_times_pipeline = np.random.randint(1, 6, num_special_orders)
special_release_times = np.random.randint(0, 20, num_special_orders)
special_due_times = special_release_times + np.random.randint(20, 40, num_special_orders)
special_profits = np.random.randint(3000, 5000, num_special_orders)


# 普通订单参数
normal_processing_times_body = np.random.randint(1, 6, (num_machines_body, num_normal_orders))
normal_processing_times_cabinet = np.random.randint(1, 6, (num_machines_cabinet, num_normal_orders))
normal_processing_times_pipeline = np.random.randint(1, 6, num_normal_orders)
normal_profits = np.random.randint(1000, 3000, num_normal_orders)

AC = 100


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
    "AC": AC
}


# 提取特殊订单和普通订单的参数
special_orders = orders_parameters["special"]
normal_orders = orders_parameters["normal"]
AC = orders_parameters["AC"]

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

print("订单信息：", orders_parameters)


#初始化决策变量矩阵
X = np.zeros((num_positions, num_orders), dtype=int) #本体顺序
Y = np.zeros((num_positions, num_orders), dtype=int) #电柜顺序

Z = np.zeros((num_positions, num_orders), dtype=int) #管线包顺序
y = np.zeros(num_special_orders, dtype=int)          #特殊订单是否被选择，y[i] = 1 表示特殊订单 i 被选择；否则为 0

# 定义适应度函数
#def fitness_function(schedule):
#    for i in range(num_machines_body):
