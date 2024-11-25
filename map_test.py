import jax
import jax.numpy as jnp

def check_numbers_exist(matrix, target_numbers):
    """
    检查矩阵中是否存在指定的数字
    
    参数:
    matrix: JAX numpy 矩阵
    target_numbers: 要检查的数字列表
    
    返回:
    存在的数字列表
    """
    # 将目标数字转换为 JAX 数组
    target_numbers = jnp.array(target_numbers)
    
    # 使用 vmap 展平矩阵并检查每个数字是否存在
    def check_number(num):
        return jnp.any(matrix == num)
    
    # 使用 vmap 并行检查每个目标数字
    exists_vector = jax.vmap(check_number)(target_numbers)
    
    # 根据存在性过滤数字
    existing_numbers = target_numbers[exists_vector]
    
    return existing_numbers

# 示例矩阵
matrix = jnp.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [25, 26, 27, 28]
])

# 目标数字
target_numbers = [91, 92, 3]

# 检查数字是否存在
result = check_numbers_exist(matrix, target_numbers)
print("存在的数字:", result)