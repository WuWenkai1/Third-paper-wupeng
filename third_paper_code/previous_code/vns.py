from parameter import num_special_orders, num_machines_body, num_machines_cabinet,processing_times_body,processing_times_cabinet,processing_times_pipeline,release_times_special,due_times_special,profits_special,AC2
import numpy as np


# 计算总的加工时长
total_processing_time_single = np.zeros(num_special_orders)  # 初始化特殊订单总加工时长矩阵

for j in range(num_special_orders):
    total_processing_time_body = np.sum(processing_times_body[:, j])
    total_processing_time_cabinet = np.sum(processing_times_cabinet[:, j])
    total_processing_time_single[j] = max(total_processing_time_body, total_processing_time_cabinet) + \
                                      processing_times_pipeline[j]


# 定义适应度函数
def fitness_function(schedule, selected_orders):
    # 初始化每台机器的开始时间和完成时间
    start_times_body = np.zeros((num_machines_body, num_special_orders))
    completion_times_body = np.zeros((num_machines_body, num_special_orders))
    start_times_cabinet = np.zeros((num_machines_cabinet, num_special_orders))
    completion_times_cabinet = np.zeros((num_machines_cabinet, num_special_orders))
    start_times_pipeline = np.zeros(num_special_orders)
    completion_times_pipeline = np.zeros(num_special_orders)


    remember = []

    for order_sequence in range(len(schedule)):
        order_index = schedule[order_sequence]
        if selected_orders[order_index] == 0:
            continue

        # 获取订单的释放时间和交期
        release_time = release_times_special[order_index]

        # 计算本体组装的开始时间和完成时间
        for i in range(num_machines_body):
            if i == 0:
                start_times_body[i, order_index] = release_time if order_sequence == 0 else max(
                    completion_times_body[i, schedule[order_sequence - 1]], release_time)
            else:
                start_times_body[i, order_index] = completion_times_body[
                    i - 1, order_index] if order_sequence == 0 else max(
                    completion_times_body[i - 1, order_index], completion_times_body[i, schedule[order_sequence - 1]])
            completion_times_body[i, order_index] = start_times_body[i, order_index] + processing_times_body[
                i, order_index]

        # 计算电柜组装的开始时间和完成时间
        for i in range(num_machines_cabinet):
            if i == 0:
                start_times_cabinet[i, order_index] = release_time if order_sequence == 0 else max(
                    completion_times_cabinet[i, schedule[order_sequence - 1]], release_time)
            else:
                start_times_cabinet[i, order_index] = completion_times_cabinet[
                    i - 1, order_index] if order_sequence == 0 else max(
                    completion_times_cabinet[i - 1, order_index],
                    completion_times_cabinet[i, schedule[order_sequence - 1]])
            completion_times_cabinet[i, order_index] = start_times_cabinet[i, order_index] + processing_times_cabinet[
                i, order_index]

        # 计算管道包装的开始时间和完成时间
        arrival_time = max(completion_times_body[:, order_index].max(),
                           completion_times_cabinet[:, order_index].max())
        if order_sequence == 0:
            start_times_pipeline[order_index] = arrival_time
        else:
            start_times_pipeline[order_index] = max(arrival_time,
                                                    completion_times_pipeline[schedule[order_sequence - 1]])
        completion_times_pipeline[order_index] = start_times_pipeline[order_index] + processing_times_pipeline[
            order_index]
        remember.append(completion_times_pipeline[order_index])

    return completion_times_pipeline, remember


def profit_cost_function(schedule, selected_orders):
    total_cost = AC2 * (max(release_times_special[selected_orders]) - min(release_times_special[selected_orders]))
    single_profit = []
    for i in range(len(schedule)):
        if selected_orders[schedule[i]] == 0:
            single_profit.append(0)
            continue
        single_cost = (total_processing_time_single[schedule[i]] / np.sum(
            total_processing_time_single[selected_orders])) * total_cost
        single_profit_1 = profits_special[schedule[i]] - single_cost
        single_profit.append(single_profit_1)

    return single_profit, total_cost


# 邻域搜索算法
def neighborhood_search(initial_schedule, max_iterations=1000):
    current_schedule = initial_schedule
    best_schedule = current_schedule
    best_fitness = -np.inf

    # 初始化选择决策变量
    selected_orders = np.random.randint(2, size=num_special_orders)

    for iteration in range(max_iterations):
        neighbors = generate_neighbors(current_schedule)
        for neighbor in neighbors:
            completion_times_pipeline, remember = fitness_function(neighbor, selected_orders)
            single_profits, total_cost = profit_cost_function(neighbor, selected_orders)

            # 检查是否所有被选择的订单都在交期内
            if all(completion_times_pipeline[order] <= due_times_special[order] for order in neighbor if
                   selected_orders[order] == 1):
                total_profit = sum(single_profits)
                if total_profit > best_fitness:
                    best_fitness = total_profit
                    best_schedule = neighbor
                    current_schedule = neighbor

    return best_schedule, best_fitness, selected_orders


# 生成邻域
def generate_neighbors(schedule):
    neighbors = []
    for i in range(len(schedule)):
        for j in range(i + 1, len(schedule)):
            neighbor = schedule.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors


# 调用函数
schedule_1 = sorted(range(len(due_times_special)), key=lambda k: due_times_special[k])

best_schedule, best_fitness, selected_orders = neighborhood_search(schedule_1)

# 打印结果
completion_times_pipeline, _ = fitness_function(best_schedule, selected_orders)
single_profits, _ = profit_cost_function(best_schedule, selected_orders)
for i in range(len(best_schedule)):
    order = best_schedule[i]
    if selected_orders[order] == 1:
        print(f"order {order}, release_time: {release_times_special[order]}, end time: {completion_times_pipeline[order]}, profit: {single_profits[i]},due_time:{due_times_special[order]}")

total_profit = sum(single_profits)
print("Total profit:", total_profit)
print("Best schedule:", best_schedule)

