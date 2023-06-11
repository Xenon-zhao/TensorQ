import numpy as np                                          # 导入numpy库并简写为np
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ   # 导入量子门H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit     # 导入Circuit模块，用于搭建量子线路

CIRCUIT = Circuit()                              # 初始化量子线路
CIRCUIT += H.on(0)                               # H门作用在第0位量子比特