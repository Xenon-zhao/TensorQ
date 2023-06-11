import numpy as np                                          # 导入numpy库并简写为np
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ, CNOT, CNOTGate, FSim   # 导入量子门H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit     # 导入Circuit模块，用于搭建量子线路

n = 35
CIRCUIT = Circuit()                              # 初始化量子线路
CIRCUIT += H.on(0)                               # H门作用在第0位量子比特
for i in range(n-1):
    CIRCUIT += CNOTGate().on(obj_qubits=i+1, ctrl_qubits=i)