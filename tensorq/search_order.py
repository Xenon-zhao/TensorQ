from .load_circuits import QuantumCircuit
from artensor import (
    find_order,
    contraction_scheme_sparse,
)
from copy import deepcopy
import numpy as np
import torch
from os.path import exists, dirname, abspath
import sys

def read_samples(filename):
    import os
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

def search_order(n = 30, m = 14, device = 'cuda', sc_target = 24, seed = 0,
    bitstrings_txt = None,
    max_bitstrings = 1, save_scheme = False, qc = None):
    """
    Search contraction order.

    Args:
        n (int, optional): the number of qubit, Default: ``30``, 
        m (int, optional): the deepth of gates, Default: ``14``, 
        device (strings, optional): device to calculate, 
            'cuda' for GPU, 'cpu' for CPU,
            Default: ``cuda`` , 
        sc_target (int, optional): target space complexity equal the memery of device (GB), 
            Default: ``24``, 
        seed (int, optional): random seed number, Default: ``24``,
        bitstrings_txt (strings, optional): a file has the bitstrings to calculate amplitude,
            Default: ``None``,
        max_bitstrings (int, optional): max number of bitstings to calculate amplitude,
            Default: ``1``
        save_scheme (bool, optional): whether or not save scheme,
            Default: ``False``
        qc (QuantumCircuit, optional):  the quantum circuit,
            Default: ``None``

    Returns:
        - **result** (tuple), a tuple include the date to contract:
            (tensors_save, scheme_sparsestate, slicing_indices, bitstrings_sorted)
            tensors_save (list): numerical tensors of the tensor network,
            scheme_sparsestate (list): list of contraction step,
            slicing_indices (dict): {tensor id: sliced indices},
            bitstrings_sorted (list): the bitstings to calculate amplitude
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    sc_target = 30 + int(np.log2(sc_target/24))
    if exists(sys.path[0] + "/scheme_n"+str(n)+"_m"+str(m)+".pt"):
        return
    if qc == None:
        qc = QuantumCircuit(n=n) # m, seq=seq
    edges = []
    for i in range(len(qc.neighbors)):
        for j in qc.neighbors[i]:
            if i < j:
                edges.append((i, j))
    neighbors = list(qc.neighbors)
    final_qubits = set(range(len(neighbors) - n, len(neighbors)))
    bond_dims = {i:2.0 for i in range(len(edges))}

    bitstrings_txt = sys.path[0] + "/" + bitstrings_txt
    if exists(bitstrings_txt):
        data = read_samples(bitstrings_txt)
        bitstrings = [data[i][0][0:n] for i in range(max_bitstrings)]
        if len(data[0])>1:
            amplitude_google = np.array([data[i][1] for i in range(max_bitstrings)])
    else:
        bitstrings = []
        for i in range(2**n):
            bitstrings.append(np.binary_repr(i,n))
    tensor_bonds = {
        i:[edges.index((min(i, j), max(i, j))) for j in neighbors[i]] 
        for i in range(len(neighbors))
    } # now all tensors will be included


    order_slicing, slicing_bonds, ctree_new = find_order(
        tensor_bonds, bond_dims, final_qubits, seed, max_bitstrings, 
        sc_target=sc_target, trials=5, iters=10, slicing_repeat=1, # iters一般设成50，slicing_repeat一般设成4，trials是用的线程数，betas是运行的逆温度
        betas=np.linspace(3.0, 21.0, 61), alpha = 10 #峰值性能/带宽
    )

    # tensors = []
    # for x in range(len(qc.tensors)):
    #     if x not in final_qubits:
    #         tensors.append(qc.tensors[x].to(device))

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
    if save_scheme:
        torch.save(result, sys.path[0] + "/scheme_n"+str(n)+"_m"+str(m)+".pt")
    print("time complexity (tc,log10), space complexity (sc,log2), memory complexity (mc) = ",ctree_new.tree_complexity())
    return result

