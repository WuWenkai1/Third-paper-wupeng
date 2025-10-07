import numpy as np

# 定义订单数量
num_special_orders = 30  # 特殊订单数量
num_normal_orders = 20   # 普通订单数量
num_orders = num_special_orders + num_normal_orders
num_positions = num_orders  # 假设每个订单在每个机器上都有一个位置

# 初始化决策变量矩阵
X = np.zeros((num_orders, num_positions), dtype=int)
Y = np.zeros((num_orders, num_positions), dtype=int)
Z = np.zeros((num_orders, num_positions), dtype=int)

# 示例：假设我们有一个初始调度
# 假设顺序为 0, 1, 2, ..., num_orders-1
schedule = list(range(num_orders))

# 初始化决策变量矩阵
for j, i in enumerate(schedule):
    X[i][j] = 1  # 订单 i 在本体组装机器上的位置 j
    Y[i][j] = 1  # 订单 i 在电柜组装机器上的位置 j
    Z[i][j] = 1  # 订单 i 在管道包装上的位置 j

# 打印决策变量矩阵
print("决策变量矩阵 X:")
print(X)
print("决策变量矩阵 Y:")
print(Y)
print("决策变量矩阵 Z:")
print(Z)

# 假设传给元启发式算法的 schedule 是由 X, Y, Z 决策变量构成的一个字典
meta_heuristic_schedule = {"X": X, "Y": Y, "Z": Z}

# 定义适应度函数
def fitness_function(meta_heuristic_schedule):
    X = meta_heuristic_schedule["X"]
    Y = meta_heuristic_schedule["Y"]
    Z = meta_heuristic_schedule["Z"]

    # 您的适应度计算逻辑...
    # 初始化每台机器的开始时间和完成时间
    num_orders = X.shape[0]
    start_times_body = np.zeros(num_orders)
    completion_times_body = np.zeros(num_orders)
    start_times_cabinet = np.zeros(num_orders)
    completion_times_cabinet = np.zeros(num_orders)
    start_times_pipeline = np.zeros(num_orders)
    completion_times_pipeline = np.zeros(num_orders)

    for order in range(num_orders):
        body_position = np.argmax(X[order])
        cabinet_position = np.argmax(Y[order])
        pipeline_position = np.argmax(Z[order])

        # 计算每个订单的加工时间和完成时间
        # (此处需要根据实际加工时间逻辑进行计算)
        start_times_body[order] = body_position  # 示例值
        completion_times_body[order] = start_times_body[order] + 1  # 示例值
        start_times_cabinet[order] = cabinet_position  # 示例值
        completion_times_cabinet[order] = start_times_cabinet[order] + 1  # 示例值
        start_times_pipeline[order] = pipeline_position  # 示例值
        completion_times_pipeline[order] = start_times_pipeline[order] + 1  # 示例值

    return completion_times_pipeline

# 调用适应度函数
completion_times = fitness_function(meta_heuristic_schedule)
print("Completion times:", completion_times)
