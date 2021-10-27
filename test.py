import numpy as np

list = [(1, 2), (2, 4), (5, 6)]
line_k = 1
line_b = 4
points = np.array(list)
value = -points[:, 0] + line_k * points[:, 1] + line_b

print(points)
print(value)