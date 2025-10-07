import numpy as np
from numpy.matrixlib.defmatrix import matrix
import random

# 定义常量和参数
num_special_orders = 30  # 特殊订单数量
num_normal_orders = 30   # 普通订单数量
num_machines_body = 3    # 本体组装机器数量
num_machines_cabinet = 5 # 电柜组装机器数量

# 随机生成参数
np.random.seed(12)

#定义订单参数
special_order = []
for i in range(num_special_orders):
    id = i + 1
    processing_body = np.random.randint(1, 6, (num_machines_body))
    processing_cabinet = np.random.randint(1, 6, (num_machines_cabinet))
    processing_pipeline = random.randint(1, 6)
    release_times_special = random.randint(0, 20)
    due_time_special = release_times_special + random.randint(20, 40)
    special_order.append({id: {
        "id": id,
        "pb": processing_body,
        "pc": processing_cabinet,
        "pp": processing_pipeline,
        "r": release_times_special,
        "d": due_time_special,
    }})

normal_order = []
for i in range(num_normal_orders):
    id = i + 1 + num_special_orders
    processing_body = np.random.randint(1, 5, (num_machines_body))
    processing_cabinet = np.random.randint(1, 5, (num_machines_cabinet))
    processing_pipeline = random.randint(1, 5)
    release_times_normal = 0
    due_time_normal = 1e6
    normal_order.append({id: {
        "id": id,
        "pb": processing_body,
        "pc": processing_cabinet,
        "pp": processing_pipeline,
        "r": release_times_normal,
        "d": due_time_normal,
    }})
AC = 1000
orders = special_order + normal_order
print(orders)

a = np.random.uniform(0,1)
b = a - 1


# 定义适应度函数及约束
#def fitness_function(X:list, Y:list, Z:list, y:matrix, order: dict ):
#    total_profit = 0
#    total_cost = 0



