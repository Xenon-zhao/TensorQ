import cirq # 0.7.0版本
import numpy as np

# 比特的网格坐标(QUBIT_ORDER是tensorq需要读取的比特变量)
n = 35
qubits = [cirq.GridQubit(0, i) for i in range(n)]
QUBIT_ORDER = qubits

# 量子线路，注意不需要'测量操作'(CIRCUIT是tensorq需要读取的线路变量)
CIRCUIT = cirq.Circuit(
    moments = [
        cirq.Moment([cirq.H(qubits[0])]),
        (cirq.Moment([cirq.CNOT(qubits[i], qubits[i+1])]) for i in range(n-1))
    ]
)