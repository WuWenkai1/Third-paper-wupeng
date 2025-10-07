import numpy as np

# 定义常量和参数
num_orders = 10  # 普通订单数量
num_special_orders = 3  # 特殊订单数量
num_total_orders = num_orders + num_special_orders  # 总订单数量

num_machines_body = 3  # 本体组装机器数量
num_machines_cabinet = 5  # 电柜组装机器数量

# 随机生成参数
np.random.seed(42)
processing_times_body = np.random.randint(1, 6, (num_machines_body, num_total_orders))
processing_times_cabinet = np.random.randint(1, 6, (num_machines_cabinet, num_total_orders))
processing_times_pipeline = np.random.randint(1, 6, num_total_orders)

release_times_special = np.random.randint(0, 10, num_special_orders)
due_times_special = np.random.randint(20, 30, num_special_orders)
profits_special = np.random.randint(50, 100, num_special_orders)
profits_common = np.random.randint(20, 50, num_orders)
processing_costs = {'body': 2, 'cabinet': 3, 'pipeline': 1}

# 定义适应度函数
def fitness_function(schedule):
    total_profit = 0
    total_cost = 0

    # 初始化每台机器的完成时间
    completion_times_body = np.zeros((num_machines_body, num_total_orders))
    completion_times_cabinet = np.zeros((num_machines_cabinet, num_total_orders))
    completion_times_pipeline = np.zeros(num_total_orders)

    for order in schedule:
        order_index, machine_body, machine_cabinet = order

        # 获取订单的释放时间和交期
        if order_index < num_special_orders:
            release_time = release_times_special[order_index]
            due_time = due_times_special[order_index]
            profit = profits_special[order_index]
        else:
            release_time = 0
            due_time = np.inf
            profit = profits_common[order_index - num_special_orders]

        # 计算本体组装的完成时间
        if order_index == 0:
            start_time_body = release_time
        else:
            start_time_body = max(completion_times_body[machine_body, :order_index].max(), release_time)
        completion_time_body = start_time_body + processing_times_body[machine_body, order_index]
        completion_times_body[machine_body, order_index] = completion_time_body

        # 计算电柜组装的完成时间
        start_time_cabinet = completion_time_body
        completion_time_cabinet = start_time_cabinet + processing_times_cabinet[machine_cabinet, order_index]
        completion_times_cabinet[machine_cabinet, order_index] = completion_time_cabinet

        # 计算管道包装的完成时间
        start_time_pipeline = completion_time_cabinet
        completion_time_pipeline = start_time_pipeline + processing_times_pipeline[order_index]
        completion_times_pipeline[order_index] = completion_time_pipeline

        # 计算订单的成本
        cost = (processing_costs['body'] * processing_times_body[machine_body, order_index] +
                processing_costs['cabinet'] * processing_times_cabinet[machine_cabinet, order_index] +
                processing_costs['pipeline'] * processing_times_pipeline[order_index])

        total_cost += cost

        # 计算利润（如果超出交期，则扣除罚款）
        if completion_time_pipeline > due_time:
            profit -= (completion_time_pipeline - due_time) * 10  # 罚款，假设每超出一天扣10单位利润

        total_profit += profit

    # 计算总利润
    net_profit = total_profit - total_cost

    return net_profit

# 随机生成调度方案（初始种群）
def generate_initial_population(size):
    population = []
    for _ in range(size):
        schedule = []
        for order_index in range(num_total_orders):
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
        fitness_scores = [fitness_function(schedule) for schedule in population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

        # 选择适应度最高的个体作为父母
        parents = sorted_population[:population_size//2]

        # 交叉生成新的个体
        offspring = []
        for _ in range(population_size - len(parents)):
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]
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
    best_schedule = sorted(population, key=fitness_function, reverse=True)[0]
    return best_schedule, fitness_function(best_schedule)

# 设置参数并运行遗传算法
population_size = 50
generations = 100
mutation_rate = 0.1

best_schedule, best_fitness = genetic_algorithm(population_size, generations, mutation_rate)

print(f"Best schedule: {best_schedule}")
print(f"Best fitness: {best_fitness}")
