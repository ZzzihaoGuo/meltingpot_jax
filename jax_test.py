import jax.numpy as jnp

# 定义二维数组
array = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8],[10,11]])  # 集合1
remove = jnp.array([[3, 4], [7, 8],[2,4]])                # 集合2

# 检查 array 的每一行是否存在于 remove 中
mask = ~jnp.any(jnp.all(array[:, None, :] == remove[None, :, :], axis=-1), axis=1)

# 应用掩码，保留不在 remove 中的行
result = array[mask]

print("结果:\n", result)

