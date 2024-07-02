import numpy as np

A = np.array([
    [1, 1],
    [-1, -1],
    [1, 1],
    [-1, -1]
])

T = np.array([
    -1, -1, -1, -1
])

b = np.array([
    1.5,
    -0.5,
    -0.5,
    1.5
])

five_accept = [
    np.array([-1, -1]),
    np.array([-1, 1]),
    np.array([0, 0]),
    np.array([1, -1]),
    np.array([1, 1])
]

four_reject = [
    np.array([-1, 0]),
    np.array([0, -1]),
    np.array([0, 1]),
    np.array([1, 0])
]

print('Five accept results:')

for sample in five_accept:
    m = A @ sample.T + b
    print(f"{sample} => \t {(m) >= 0} {(T @ m + 2.5) >= 0}")
    
print('Four reject results:')

for sample in four_reject:
    m = A @ sample.T + b
    print(f"{sample} => \t {(m) >= 0} {(T @ m + 2.5) >= 0}")
