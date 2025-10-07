import gurobipy as gp
from gurobipy import GRB
import numpy as np

# 定义常量和参数
num_special_orders = 30  # 特殊订单数量
num_machines_body = 3  # 本体组装机器数量
num_machines_cabinet = 5  # 电柜组装机器数量

# 随机生成参数
np.random.seed(42)
processing_times_body = np.random.randint(1, 6, (num_machines_body, num_special_orders))
processing_times_cabinet = np.random.randint(1, 6, (num_machines_cabinet, num_special_orders))
processing_times_pipeline = np.random.randint(1, 6, num_special_orders)

release_times_special = np.random.randint(0, 50, num_special_orders)
due_times_special = np.random.randint(100, 200, num_special_orders)
profits_special = np.random.randint(3000, 5000, num_special_orders)

AC2 = 100
M = 1e6  # 设置一个超大数

# 创建模型
model = gp.Model("SpecialOrdersScheduling")

# 创建变量
x = model.addVars(num_special_orders, num_special_orders, vtype=GRB.BINARY, name="x")  # x[i, j]表示订单i是否在订单j之前
start_time = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="start_time")
start_times_body = model.addVars(num_machines_body, num_special_orders, vtype=GRB.CONTINUOUS, name="start_times_body")
completion_times_body = model.addVars(num_machines_body, num_special_orders, vtype=GRB.CONTINUOUS, name="completion_times_body")
start_times_cabinet = model.addVars(num_machines_cabinet, num_special_orders, vtype=GRB.CONTINUOUS, name="start_times_cabinet")
completion_times_cabinet = model.addVars(num_machines_cabinet, num_special_orders, vtype=GRB.CONTINUOUS, name="completion_times_cabinet")
start_times_pipeline = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="start_times_pipeline")
completion_times_pipeline = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="completion_times_pipeline")
u = model.addVars(num_special_orders, vtype=GRB.BINARY, name="u")  # u 表示是否选择该特殊订单

# 辅助变量
max_completion_body = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="max_completion_body")
max_completion_cabinet = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="max_completion_cabinet")

# 目标函数：最大化特殊订单的总利润
model.setObjective(gp.quicksum(u[j] * profits_special[j] for j in range(num_special_orders)) -
                   gp.quicksum(AC2 * (completion_times_pipeline[j] - start_time[j]) for j in range(num_special_orders)),
                   GRB.MAXIMIZE)

# 约束条件

# 本体组装的时间约束
for j in range(num_special_orders):
    for i in range(num_machines_body):
        if i == 0:
            model.addConstr(start_times_body[i, j] >= release_times_special[j])
        else:
            model.addConstr(start_times_body[i, j] >= completion_times_body[i-1, j])
        model.addConstr(completion_times_body[i, j] == start_times_body[i, j] + processing_times_body[i, j])

# 电柜组装的时间约束
for j in range(num_special_orders):
    for i in range(num_machines_cabinet):
        if i == 0:
            model.addConstr(start_times_cabinet[i, j] >= release_times_special[j])
        else:
            model.addConstr(start_times_cabinet[i, j] >= completion_times_cabinet[i-1, j])
        model.addConstr(completion_times_cabinet[i, j] == start_times_cabinet[i, j] + processing_times_cabinet[i, j])

# 最大完成时间约束
for j in range(num_special_orders):
    for i in range(num_machines_body):
        model.addConstr(max_completion_body[j] >= completion_times_body[i, j])
    for i in range(num_machines_cabinet):
        model.addConstr(max_completion_cabinet[j] >= completion_times_cabinet[i, j])

# 管道包装的时间约束
for j in range(num_special_orders):
    model.addConstr(start_times_pipeline[j] >= max_completion_body[j])
    model.addConstr(start_times_pipeline[j] >= max_completion_cabinet[j])
    model.addConstr(completion_times_pipeline[j] == start_times_pipeline[j] + processing_times_pipeline[j])

# 新增约束：订单的顺序约束
# 本体订单顺序约束
for i in range(num_machines_body):
    for k in range(num_special_orders):
        for v in range(num_special_orders):
            if k != v:
                model.addConstr(start_times_body[i, k] >= completion_times_body[i, v] - M * (1 - x[k, v]))
                model.addConstr(x[k, v] + x[v, k] <= 1)  # 防止循环依赖

# 电柜订单顺序约束
for i in range(num_machines_cabinet):
    for k in range(num_special_orders):
        for v in range(num_special_orders):
            if k != v:
                model.addConstr(start_times_cabinet[i, k] >= completion_times_cabinet[i, v] - M * (1 - x[k, v]))
                model.addConstr(x[k, v] + x[v, k] <= 1)  # 防止循环依赖

# 管线包订单顺序约束
for k in range(num_special_orders):
    for v in range(num_special_orders):
        if k != v:
            model.addConstr(start_times_pipeline[k] >= completion_times_pipeline[v] - M * (1 - x[k, v]))
            model.addConstr(x[k, v] + x[v, k] <= 1)  # 防止循环依赖

# 完工时间不能超过交期
for j in range(num_special_orders):
    model.addConstr(completion_times_pipeline[j] <= due_times_special[j] + (1 - u[j]) * M)

# 订单最初开始时间计算
for j in range(num_special_orders):
    model.addConstr(start_time[j] <= start_times_body[0, j] * u[j])
    model.addConstr(start_time[j] <= start_times_cabinet[0, j] * u[j])

# 每个订单只能有一个前序和后序
for k in range(num_special_orders):
    model.addConstr(gp.quicksum(x[k, v] for v in range(num_special_orders) if k != v) == u[k])
    model.addConstr(gp.quicksum(x[v, k] for v in range(num_special_orders) if v != k) == u[k])

# 求解模型

model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal}")
    for j in range(num_special_orders):
        if u[j].x > 0.5:
            print(f"Order {j}:")
            print(f"  Start time body: {[start_times_body[i, j].x for i in range(num_machines_body)]}")
            print(f"  Completion time body: {[completion_times_body[i, j].x for i in range(num_machines_body)]}")
            print(f"  Start time cabinet: {[start_times_cabinet[i, j].x for i in range(num_machines_cabinet)]}")
            print(f"  Completion time cabinet: {[completion_times_cabinet[i, j].x for i in range(num_machines_cabinet)]}")
            print(f"  Start time pipeline: {start_times_pipeline[j].x}")
            print(f"  Completion time pipeline: {completion_times_pipeline[j].x}")
else:
    print("No optimal solution found.")
