import numpy as np
import pdb
def is_row_vector(x):
    return (x.size == 1 and x.ndim == 1) or (np.ndim(x) == 1 and x.shape[0] == 1)

# 调用示例
x = np.array([1, 2, 3])  # 列向量
pdb.set_trace()
if is_row_vector(x):
    print("是row向量")
else:
    print("不是row向量")