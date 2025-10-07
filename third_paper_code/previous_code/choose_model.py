import numpy as np
from gurobipy import Model, GRB, quicksum

# 定义订单数量
num_special_orders = 10
num_normal_orders = 30
total_orders = num_special_orders + num_normal_orders

# 随机生成订单数据
np.random.seed(14)  # 为了重现结果，设置随机种子

special_orders = [{
    'id': i,
    'profit': np.random.randint(1000, 2000),
    'release_time': np.random.randint(1, 50),
    'due_time': np.random.randint(60, 100),
    'process_times_body': np.random.randint(5, 15, 3).tolist(),
    'process_times_cabinet': np.random.randint(5, 15, 5).tolist(),
    'process_time_pack': np.random.randint(5, 15)
} for i in range(num_special_orders)]

normal_orders = [{
    'id': i + num_special_orders,
    'profit': np.random.randint(100, 500),
    'release_time': None,
    'due_time': None,
    'process_times_body': np.random.randint(5, 15, 3).tolist(),
    'process_times_cabinet': np.random.randint(5, 15, 5).tolist(),
    'process_time_pack': np.random.randint(5, 15)
} for i in range(num_normal_orders)]

orders = special_orders + normal_orders

# 生产周期和成本参数
T = 200
AC1 = 50
AC2 = 100
M = 100000  # 一个非常大的数


def calculate_profit(solution, orders, AC1, AC2):
    total_profit = 0
    current_time_body = [0] * 3  # 本体组装组三个操作的当前时间
    current_time_cabinet = [0] * 5  # 电柜组装组五个操作的当前时间
    current_time_pack = 0  # 管线包组装的当前时间

    for order_id in solution:
        order = special_orders[order_id]
        if 'release_time' in order:
            current_time_body[0] = max(current_time_body[0], order['release_time'])

        # 本体组装时间
        for p in range(3):
            start_time = current_time_body[p]
            finish_time = start_time + order['process_times_body'][p]
            current_time_body[p] = finish_time
            if p < 2:
                current_time_body[p + 1] = max(current_time_body[p + 1], finish_time)

        # 电柜组装时间
        for q in range(5):
            start_time = current_time_cabinet[q]
            finish_time = start_time + order['process_times_cabinet'][q]
            current_time_cabinet[q] = finish_time
            if q < 4:
                current_time_cabinet[q + 1] = max(current_time_cabinet[q + 1], finish_time)

        # 管线包组装时间
        start_time_pack = max(current_time_body[-1], current_time_cabinet[-1])
        start_time_pack = max(start_time_pack, current_time_pack)
        finish_time_pack = start_time_pack + order['process_time_pack']
        current_time_pack = finish_time_pack

        C_max = max(C_max, finish_time_pack)
        total_profit += order['profit']

    total_cost = AC1 * min(C_max, T) + AC2 * max(0, C_max - T)
    total_profit -= total_cost

    # 检查特殊订单的交货期约束
    for order in orders:
        if 'due_time' in order and order['due_time'] is not None and current_time_pack > order['due_time']:
            return -float('inf')  # 如果违反交货期约束，解无效

    return total_profit


def fitness(solution, orders, T, AC1, AC2):
    return calculate_profit(solution, orders, T, AC1, AC2)


def generate_initial_solution(orders):
    # 将订单按释放时间和截止时间排序，确保特殊订单不违反交货期约束
    special_orders = [order for order in orders if order['release_time'] is not None]
    normal_orders = [order for order in orders if order['release_time'] is None]

    sorted_special_orders = sorted(special_orders, key=lambda x: (x['release_time'], x['due_time']))
    sorted_normal_orders = sorted(normal_orders, key=lambda x: x['profit'], reverse=True)

    initial_solution = sorted_special_orders + sorted_normal_orders
    return [order['id'] for order in initial_solution]


# 使用Gurobi求解问题
def solve_with_gurobi(special_orders):
    num_special_orders = len(special_orders)

    model = Model("MaximizeSpecialOrderProfit")

    # 添加变量
    x = model.addVars(num_special_orders, vtype=GRB.BINARY, name="x")
    start_times_body = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="start_times_body")
    start_times_cabinet = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="start_times_cabinet")
    finish_times_body = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="finish_times_body")
    finish_times_cabinet = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="finish_times_cabinet")
    start_times_pack = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="start_times_pack")
    finish_times_pack = model.addVars(num_special_orders, vtype=GRB.CONTINUOUS, name="finish_times_pack")

    # 设置目标函数：最大化利润
    model.setObjective(quicksum(special_orders[i]['profit'] * x[i] for i in range(num_special_orders)), GRB.MAXIMIZE)

    # 添加约束条件
    for i in range(num_special_orders):
        order = special_orders[i]

        # 本体组装组装约束
        model.addConstr(start_times_body[i] >= order['release_time'] * x[i], name=f"release_time_body_constr_{i}")
        model.addConstr(finish_times_body[i] == start_times_body[i] + sum(order['process_times_body']) * x[i],
                        name=f"finish_time_body_constr_{i}")

        # 电柜组装组装约束
        model.addConstr(start_times_cabinet[i] >= order['release_time'] * x[i], name=f"release_time_cabinet_constr_{i}")
        model.addConstr(finish_times_cabinet[i] == start_times_cabinet[i] + sum(order['process_times_cabinet']) * x[i],
                        name=f"finish_time_cabinet_constr_{i}")

        # 管线包装配约束
        model.addConstr(start_times_pack[i] >= finish_times_body[i], name=f"start_time_pack_body_constr_{i}")
        model.addConstr(start_times_pack[i] >= finish_times_cabinet[i], name=f"start_time_pack_cabinet_constr_{i}")
        model.addConstr(finish_times_pack[i] == start_times_pack[i] + order['process_time_pack'] * x[i],
                        name=f"finish_time_pack_constr_{i}")

        # 订单的完成时间必须小于等于交期
        model.addConstr(finish_times_pack[i] <= order['due_time'] + (1 - x[i]) * 1e6, name=f"due_time_constr_{i}")

        # 订单的完成时间必须小于等于生产周期
        model.addConstr(finish_times_pack[i] <= T + (1 - x[i]) * 1e6, name=f"T_constr_{i}")

    # 添加非重叠约束
    for i in range(num_special_orders):
        for j in range(i + 1, num_special_orders):
            model.addConstr(
                (start_times_body[j] >= finish_times_body[i]) + (start_times_body[i] >= finish_times_body[j]) +
                (start_times_cabinet[j] >= finish_times_cabinet[i]) + (
                            start_times_cabinet[i] >= finish_times_cabinet[j]) +
                (start_times_pack[j] >= finish_times_pack[i]) + (start_times_pack[i] >= finish_times_pack[j]) >= 1,
                name=f"no_overlap_constr_{i}_{j}")

    # 求解模型
    model.optimize()

    # 打印结果
    print("选出的特殊订单:")
    for i in range(num_special_orders):
        if x[i].X > 0.5:
            print(
                f"订单ID: {special_orders[i]['id']}, 收入: {special_orders[i]['revenue']}, 利润: {special_orders[i]['profit']}, 释放时间: {special_orders[i]['release_time']}, 交期: {special_orders[i]['due_time']}, 本体开始时间: {start_times_body[i].X}, 本体完工时间: {finish_times_body[i].X}, 电柜开始时间: {start_times_cabinet[i].X}, 电柜完工时间: {finish_times_cabinet[i].X}, 包装开始时间: {start_times_pack[i].X}, 包装完工时间: {finish_times_pack[i].X}")


# 使用Gurobi求解
solve_with_gurobi(special_orders)
