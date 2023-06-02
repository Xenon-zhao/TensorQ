from load_circuits import QuantumCircuit
from artensor import (
    find_order, 
    contraction_scheme_sparse,
)
from copy import deepcopy
import numpy as np
import torch
from os.path import exists, dirname, abspath

def read_samples(filename):
    import os
    filename = abspath(dirname(__file__)) + '/' + filename
    if os.path.exists(filename):
        samples_data = []
        with open(filename, 'r') as f:
            l = f.readlines()
        f.close()
        if ' ' in l[0]:
            for line in l:
                ll = line.split()
                samples_data.append((ll[0], float(ll[1]) + 1j*float(ll[2])))
        else:
            for line in l:
                ll = line
                samples_data.append([ll])
        return samples_data
    else:
        raise ValueError("{} does not exist".format(filename))

def search_order(n = 30, m = 14, seq = 'EFGH', device = 'cuda', sc_target = 30, seed = 0,
    bitstrings_txt = 'amplitudes_n30_m14_s0_e0_pEFGH_10000.txt',
    max_bitstrings = 1_000):
    torch.backends.cuda.matmul.allow_tf32 = False
    if exists(abspath(dirname(__file__)) + "/scheme_n"+str(n)+"_m"+str(m)+".pt"):
        return

    qc = QuantumCircuit(n, m, seq=seq)
    edges = []
    for i in range(len(qc.neighbors)):
        for j in qc.neighbors[i]:
            if i < j:
                edges.append((i, j))
    neighbors = list(qc.neighbors)
    final_qubits = set(range(len(neighbors) - n, len(neighbors)))
    bond_dims = {i:2.0 for i in range(len(edges))}

    data = read_samples(bitstrings_txt)
    bitstrings = [data[i][0][0:n] for i in range(max_bitstrings)]
    if len(data[0])>1:
        amplitude_google = np.array([data[i][1] for i in range(max_bitstrings)])
    tensor_bonds = {
        i:[edges.index((min(i, j), max(i, j))) for j in neighbors[i]] 
        for i in range(len(neighbors))
    } # now all tensors will be included


    order_slicing, slicing_bonds, ctree_new = find_order(
        tensor_bonds, bond_dims, final_qubits, seed, max_bitstrings, 
        sc_target=sc_target, trials=5, iters=10, slicing_repeat=1, # iters一般设成50，slicing_repeat一般设成4，trials是用的线程数，betas是运行的逆温度
        betas=np.linspace(3.0, 21.0, 61), alpha = 10 #峰值性能/带宽
    )

    tensors = []
    for x in range(len(qc.tensors)):
        if x not in final_qubits:
            tensors.append(qc.tensors[x].to(device))

    scheme_sparsestate, _, bitstrings_sorted = contraction_scheme_sparse(
        ctree_new, bitstrings, sc_target=sc_target)

    slicing_edges = [edges[i] for i in slicing_bonds]
    slicing_indices = {}.fromkeys(slicing_edges)
    tensors = []
    for x in range(len(qc.tensors)):
        if x in final_qubits:
            tensors.append(
                torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device=device)
            )
        else:
            tensors.append(qc.tensors[x].to(device))

    tensors_save = [tensor.to('cpu') for tensor in tensors]

    neighbors_copy = deepcopy(neighbors)
    for x, y in slicing_edges:
        idxi_j = neighbors_copy[x].index(y)
        idxj_i = neighbors_copy[y].index(x)
        neighbors_copy[x].pop(idxi_j)
        neighbors_copy[y].pop(idxj_i)
        slicing_indices[(x, y)] = (idxi_j, idxj_i)

    result = (tensors_save, scheme_sparsestate, slicing_indices, bitstrings_sorted)
    torch.save(result, abspath(dirname(__file__)) + "/scheme_n"+str(n)+"_m"+str(m)+".pt")
    print("时间复杂度tc, 空间复杂度sc, 内存复杂度mc = ",ctree_new.tree_complexity())

