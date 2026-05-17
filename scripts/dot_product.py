import numpy as np

from lib import weight_to_rust

vals = [
    ((3,), (3,)),
    ((2,), (2, 2)),
    ((3, 2), (2, 4)),
]
for x, y in vals:
    print(x, y)
    a = np.random.rand(*x).astype(np.float32)
    b = np.random.rand(*y).astype(np.float32)
    print(weight_to_rust("A", a))
    print(weight_to_rust("B", b))
    value = np.array(a).dot(np.array(b))
    print(weight_to_rust("EXPECT", value))
