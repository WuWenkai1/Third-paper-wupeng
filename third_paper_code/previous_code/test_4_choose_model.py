from parameter import num_special_orders, num_machines_body, num_machines_cabinet,processing_times_body,processing_times_cabinet,processing_times_pipeline,release_times_special,due_times_special,profits_special,AC2
import numpy as np

# 计算总的加工时长
total_processing_time_single = np.zeros(num_special_orders)  # 初始化特殊订单总加工时长矩阵

for j in range(num_special_orders):
    total_processing_time_body = np.sum(processing_times_body[:, j])
    total_processing_time_cabinet = np.sum(processing_times_cabinet[:, j])
    total_processing_time_single[j] = max(total_processing_time_body, total_processing_time_cabinet) + \
                                      processing_times_pipeline[j]

# 初步筛选条件
valid_indices = [i for i in range(num_special_orders) if total_processing_time_single[i] <= due_times_special[i]]

# 根据有效索引筛选订单
processing_times_body = processing_times_body[:, valid_indices]
processing_times_cabinet = processing_times_cabinet[:, valid_indices]
processing_times_pipeline = processing_times_pipeline[valid_indices]
release_times_special = release_times_special[valid_indices]
due_times_special = due_times_special[valid_indices]
profits_special = profits_special[valid_indices]
total_processing_time_single = total_processing_time_single[valid_indices]

num_special_orders = len(valid_indices)  # 更新特殊订单数量

print("选出的订单：",valid_indices)
# 定义适应度函数
def fitness_function(schedule):
    # 初始化每台机器的开始时间和完成时间
    start_times_body = np.zeros((num_machines_body, num_special_orders))
    completion_times_body = np.zeros((num_machines_body, num_special_orders))
    start_times_cabinet = np.zeros((num_machines_cabinet, num_special_orders))
    completion_times_cabinet = np.zeros((num_machines_cabinet, num_special_orders))
    start_times_pipeline = np.zeros(num_special_orders)
    completion_times_pipeline = np.zeros(num_special_orders)
    order_percent = np.zeros(num_special_orders)

    total_processing_time_all = np.sum(total_processing_time_single[schedule])

    remember = []
    order_percentages = []

    for order_sequence in range(len(schedule)):
        order_index = schedule[order_sequence]
        # 计算每个特殊订单的所占比重
        order_percent[order_index] = total_processing_time_single[order_index] / total_processing_time_all
        order_percentages.append(order_percent[order_index])

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
                    completion_times_cabinet[i - 1, order_index], completion_times_cabinet[i, schedule[order_sequence - 1]])
            completion_times_cabinet[i, order_index] = start_times_cabinet[i, order_index] + processing_times_cabinet[
                i, order_index]

        # 计算管道包装的开始时间和完成时间
        arrival_time = max(completion_times_body[:, order_index].max(),
                           completion_times_cabinet[:, order_index].max())
        if order_sequence == 0:
            start_times_pipeline[order_index] = arrival_time
        else:
            start_times_pipeline[order_index] = max(arrival_time, completion_times_pipeline[schedule[order_sequence - 1]])
        completion_times_pipeline[order_index] = start_times_pipeline[order_index] + processing_times_pipeline[
            order_index]
        remember.append(completion_times_pipeline[order_index])

    return completion_times_pipeline, order_percentages, remember

def profit_cost_function(schedule, order_percentages, remember):
    total_cost = AC2 * (max(remember) - min(release_times_special))
    single_profit = []
    for i in range(len(schedule)):
        single_cost = (total_processing_time_single[schedule[i]] / np.sum(
            total_processing_time_single[schedule])) * total_cost
        single_profit_1 = profits_special[schedule[i]] - single_cost
        single_profit.append(single_profit_1)

    return single_profit, total_cost

# 更新排序方案并计算利润(选择具体过程)
def update_schedule(schedule):
    while True:
        completion_times_pipeline, order_percentages, remember = fitness_function(schedule)
        single_profits, total_cost = profit_cost_function(schedule, order_percentages, remember)
        changes_made = False

        for idx_i in range(1, len(schedule)):
            i = schedule[idx_i]
            if completion_times_pipeline[i] > due_times_special[i]:
                previous_order = schedule[idx_i - 1]
                if single_profits[idx_i] < single_profits[idx_i - 1]:
                    print(f"Deleting order {i} because its profit is less than the previous order {previous_order}.")
                    schedule.pop(idx_i)
                    changes_made = True
                    break
                elif idx_i < len(schedule) - 1:
                    next_order = schedule[idx_i + 1]
                    if single_profits[idx_i] > (single_profits[idx_i - 1] + single_profits[idx_i + 1]):
                        print(f"Deleting previous order {previous_order} because the current order {i}'s profit is greater than the sum of the previous and next order.")
                        schedule.pop(idx_i - 1)
                        changes_made = True
                        break
                    elif completion_times_pipeline[next_order] < due_times_special[next_order]:
                        schedule.pop(idx_i - 1)
                        changes_made = True
                        break

                    elif completion_times_pipeline[next_order] > due_times_special[next_order]:
                        print(f"Deleting current order {i} because the next order {next_order} cannot be completed on time.")
                        old_schedule = schedule.copy()
                        schedule.pop(idx_i)
                        completion_times_pipeline, order_percentages, remember = fitness_function(schedule)
                        if completion_times_pipeline[schedule[idx_i]] < due_times_special[schedule[idx_i]]:
                            changes_made = True
                            break
                        else:
                            old_schedule.pop(idx_i - 1)
                            schedule = old_schedule
                            changes_made = True
                            break
                elif idx_i == len(schedule):
                    schedule.pop(idx_i - 1)
                    changes_made = True
                    break

        if not changes_made:
            break

    return schedule, single_profits, total_cost

# 调用函数
schedule_1 = sorted(range(len(due_times_special)), key=lambda k: due_times_special[k])

# 遍历 `schedule_1` 中的元素 `i`，再遍历 `i` 之后的元素 `j`
for idx_i, i in enumerate(schedule_1):
    for idx_j in range(idx_i + 1, len(schedule_1)):
        j = schedule_1[idx_j]
        if release_times_special[i] >= release_times_special[j] and release_times_special[i] - release_times_special[j] >= due_times_special[j] - due_times_special[i]:
            # 交换 i 和 j 的位置
            schedule_1[idx_i], schedule_1[idx_j] = schedule_1[idx_j], schedule_1[idx_i]

schedule, single_profits, total_cost = update_schedule(schedule_1)

# 打印结果
completion_times_pipeline, _, _ = fitness_function(schedule)
for i in range(len(schedule)):
    order = schedule[i]
    print(f"order {order}, release_time: {release_times_special[order]}, end time: {completion_times_pipeline[order]}, profit: {single_profits[i]}，due_time:{due_times_special[order]}")

total_profit = sum(single_profits)
print("Total profit:", total_profit)
print("Updated schedule:", schedule)
