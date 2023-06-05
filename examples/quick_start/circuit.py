import cirq # 0.7.0版本
import numpy as np
# 比特的网格坐标
QUBIT_ORDER = [
    cirq.GridQubit(0, 0),
]

# 量子线路，注意不需要'测量操作'
CIRCUIT = cirq.Circuit(
    moments=[
        cirq.Moment(
            operations=[
                cirq.H(cirq.GridQubit(0, 0))
            ]
        ),
    ]
)