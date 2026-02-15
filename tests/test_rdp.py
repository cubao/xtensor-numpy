from pocket_numpy import rdp
import numpy as np

ret = rdp([[0, 0], [1, 1], [2, 2], [3, 3]])
assert isinstance(ret, np.ndarray)
assert ret.tolist() == [[0, 0], [3, 3]]

arr = [[0, 0], [5, 1], [10, 0]]
ret = rdp(arr, epsilon=1.0)
assert ret.tolist() == [[0, 0], [10, 0]]
ret = rdp(arr, epsilon=1.0 - 1e-6)
assert ret.tolist() == arr

arr = [[*xy, 0.0] for xy in arr]
ret = rdp(arr, epsilon=1.0)
assert ret.tolist() == [[0, 0, 0], [10, 0, 0]]
ret = rdp(arr, epsilon=1.0 - 1e-6)
assert ret.tolist() == arr