"""
import numpy as np
import random

# 定义常量和参数
num_special_orders = 30  # 特殊订单数量
num_machines_body = 3  # 本体组装机器数量
num_machines_cabinet = 5  # 电柜组装机器数量
population_size = 50  # 种群大小
generations = 100  # 迭代次数
mutation_rate = 0.1  # 变异概率

# 随机生成参数
np.random.seed(42)
processing_times_body = np.random.randint(1, 6, (num_machines_body, num_special_orders))
processing_times_cabinet = np.random.randint(1, 6, (num_machines_cabinet, num_special_orders))
processing_times_pipeline = np.random.randint(1, 6, num_special_orders)

release_times_special = np.random.randint(0, 50, num_special_orders)
due_times_special = np.random.randint(100, 200, num_special_orders)
profits_special = np.random.randint(3000, 5000, num_special_orders)

AC2 = 100

# 定义适应度函数
def fitness_function(schedule):
    total_profit = 0
    total_cost = 0

    # 初始化每台机器的开始时间和完成时间
    start_times_body = np.zeros((num_machines_body, num_special_orders))
    completion_times_body = np.zeros((num_machines_body, num_special_orders))
    start_times_cabinet = np.zeros((num_machines_cabinet, num_special_orders))
    completion_times_cabinet = np.zeros((num_machines_cabinet, num_special_orders))
    start_times_pipeline = np.zeros(num_special_orders)
    completion_times_pipeline = np.zeros(num_special_orders)

    for order in schedule:
        order_index, machine_body, machine_cabinet = order

        # 获取订单的释放时间和交期
        if order_index < num_special_orders:
            release_time = release_times_special[order_index]
            due_time = due_times_special[order_index]
            profit = profits_special[order_index]

            # 计算本体组装的开始时间和完成时间
            for i in range(num_machines_body):
                if i == 0:
                    if order_index == 0:
                        start_times_b00ody[i, order_index] = release_time
                    else:
                        start_times_body[i, order_index] = max(completion_times_body[i, order_index - 1], release_time)
                else:
                    start_times_body[i, order_index] = completion_times_body[i-1, order_index]
                completion_times_body[i, order_index] = start_times_body[i, order_index] + processing_times_body[i, order_index]

            # 计算电柜组装的开始时间和完成时间
            for i in range(num_machines_cabinet):
                if i == 0:
                    if order_index == 0:
                        start_times_cabinet[i, order_index] = release_time
                    else:
                        start_times_cabinet[i, order_index] = max(completion_times_cabinet[i, order_index - 1], release_time)
                else:
                    start_times_cabinet[i, order_index] = completion_times_cabinet[i-1, order_index]
                completion_times_cabinet[i, order_index] = start_times_cabinet[i, order_index] + processing_times_cabinet[i, order_index]

            # 计算管道包装的开始时间和完成时间
            start_times_pipeline[order_index] = max(completion_times_body[:, order_index].max(), completion_times_cabinet[:, order_index].max())
            completion_times_pipeline[order_index] = start_times_pipeline[order_index] + processing_times_pipeline[order_index]

            # 计算订单的成本
            start_time_min = min(start_times_body[:, order_index].min(), start_times_cabinet[:, order_index].min())
            cost = AC2 * (completion_times_pipeline[order_index] - start_time_min)
            total_cost += cost

            # 计算利润（如果超出交期，则扣除罚款）
            if completion_times_pipeline[order_index] > due_time:
                profit -= (completion_times_pipeline[order_index] - due_time) * 10  # 罚款，假设每超出一天扣10单位利润

            total_profit += profit

    # 计算总利润
    net_profit = total_profit - total_cost

    return net_profit, start_times_body, completion_times_body, start_times_cabinet, completion_times_cabinet, start_times_pipeline, completion_times_pipeline

# 随机生成调度方案（初始种群）
def generate_initial_population(size):
    population = []
    for _ in range(size):
        schedule = []
        for order_index in range(num_special_orders):
            machine_body = np.random.randint(0, num_machines_body)
            machine_cabinet = np.random.randint(0, num_machines_cabinet)
            schedule.append((order_index, machine_body, machine_cabinet))
        np.random.shuffle(schedule)
        population.append(schedule)
    return population

# 遗传算法求解
def genetic_algorithm(population_size, generations, mutation_rate):
    population = generate_initial_population(population_size)
    for generation in range(generations):
        # 评估适应度
        fitness_results = [fitness_function(schedule) for schedule in population]
        fitness_scores = [result[0] for result in fitness_results]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

        # 选择适应度最高的个体作为父母
        parents = sorted_population[:population_size//2]

        # 交叉生成新的个体
        offspring = []
        for _ in range(population_size - len(parents)):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            crossover_point = np.random.randint(0, len(parent1))
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)

        # 变异
        for child in offspring:
            if np.random.rand() < mutation_rate:
                mutate_index = np.random.randint(0, len(child))
                machine_body = np.random.randint(0, num_machines_body)
                machine_cabinet = np.random.randint(0, num_machines_cabinet)
                child[mutate_index] = (child[mutate_index][0], machine_body, machine_cabinet)

        # 更新种群
        population = parents + offspring

    # 返回最佳个体
    best_schedule = sorted(population, key=lambda sch: fitness_function(sch)[0], reverse=True)[0]
    best_fitness, best_start_times_body, best_completion_times_body, best_start_times_cabinet, best_completion_times_cabinet, best_start_times_pipeline, best_completion_times_pipeline = fitness_function(best_schedule)
    return best_schedule, best_fitness, best_start_times_body, best_completion_times_body, best_start_times_cabinet, best_completion_times_cabinet, best_start_times_pipeline, best_completion_times_pipeline

# 设置参数并运行遗传算法
best_schedule, best_fitness, best_start_times_body, best_completion_times_body, best_start_times_cabinet, best_completion_times_cabinet, best_start_times_pipeline, best_completion_times_pipeline = genetic_algorithm(population_size, generations, mutation_rate)

print(f"Best schedule: {best_schedule}")
print(f"Best fitness: {best_fitness}")

# 打印所有被选择的订单的具体信息
print("Selected orders details:")
for order in best_schedule:
    order_index, machine_body, machine_cabinet = order
    print(f"Order {order_index}:")
    print(f"  Machine body: {machine_body}")
    print(f"  Machine cabinet: {machine_cabinet}")
    print(f"  Start time body: {[best_start_times_body[i, order_index] for i in range(num_machines_body)]}")
    print(f"  Completion time body: {[best_completion_times_body[i, order_index] for i in range(num_machines_body)]}")
    print(f"  Start time cabinet: {[best_start_times_cabinet[i, order_index] for i in range(num_machines_cabinet)]}")
    print(f"  Completion time cabinet: {[best_completion_times_cabinet[i, order_index] for i in range(num_machines_cabinet)]}")
    print(f"  Start time pipeline: {best_start_times_pipeline[order_index]}")
    print(f"  Completion time pipeline: {best_completion_times_pipeline[order_index]}")
"""