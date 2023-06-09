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
        if ' ' in l[0]: # read file like: '100010001000000011011000101000   4.13647322e-05  -3.39767357e-05'
            for line in l:
                ll = line.split()
                samples_data.append((ll[0], float(ll[1]) + 1j*float(ll[2])))
        elif '\t' in l[0]: # read file like: '00000000000110000000001100001001011001000100100011110	(9.34221e-09,-4.77717e-10)	8.750510121618899e-17'
            for line in l:
                ll = line.split('\t')
                real, imag = map(float, ll[1].strip('()').split(','))
                samples_data.append((ll[0], complex(real, imag), float(ll[2])))
        else: # read file like: '100010001000000011011000101000'
            for line in l:
                ll = line
                samples_data.append([ll])
        return samples_data
    else:
        raise ValueError("{} does not exist".format(filename))

def search_order(n = 30, m = 14, device = 'cuda', sc_target = 24, seed = 0,
    bitstrings_txt = None,
    max_bitstrings = 1, save_scheme = False, qc = None, fname = None, 
    trials=5, iters=10, slicing_repeat=1,
    betas=np.linspace(3.0, 21.0, 61), alpha = 128): # iters一般设成50，slicing_repeat一般设成4，trials是用的线程数，betas是运行的逆温度
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
        trials (int, optional): the number of search thread, suggestive value '36',
            Default: ``5``
        iters (int, optional): the number of interation, suggestive value '50',
            Default: ``10``
        slicing_repeat (int, optional): the number of slicing repeat, suggestive value '4',
            Default: ``1``
        betas (numpy.ndarray, optional): the inverse temperature,
            Default: ``numpy.linspace(3.0, 21.0, 61)``
        alpha (int, optioanl): the ratio of floating point computing speed(TFLOPS) to memory access bandwidth(TB/s),
            '128' for Nvidia A100 SXM(TF32),  '20' for Nvidia V100S PCle(FP32)
            Default: ``128``

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
        return torch.load(sys.path[0] + "/scheme_n"+str(n)+"_m"+str(m)+".pt")
    if qc == None:
        qc = QuantumCircuit(n=n, fname=fname) # m, seq=seq
    edges = []
    for i in range(len(qc.neighbors)):
        for j in qc.neighbors[i]:
            if i < j:
                edges.append((i, j))
    neighbors = list(qc.neighbors)
    final_qubits = set(range(len(neighbors) - n, len(neighbors)))
    bond_dims = {i:2.0 for i in range(len(edges))}

    if bitstrings_txt == None:
        bitstrings = []
        for i in range(2**n):
            bitstrings.append(np.binary_repr(i,n))
    else:
        bitstrings_txt = sys.path[0] + "/" + bitstrings_txt
        if exists(bitstrings_txt):
            data = read_samples(bitstrings_txt)
            bitstrings = [data[i][0][0:n] for i in range(max_bitstrings)]
            if len(data[0])>1:
                amplitude_google = np.array([data[i][1] for i in range(max_bitstrings)])
        else:
            raise(f'Don\'t find the file:{bitstrings_txt}')
    tensor_bonds = {
        i:[edges.index((min(i, j), max(i, j))) for j in neighbors[i]] 
        for i in range(len(neighbors))
    } # now all tensors will be included


    order_slicing, slicing_bonds, ctree_new = find_order(
        tensor_bonds, bond_dims, final_qubits, seed, max_bitstrings, 
        sc_target=sc_target, trials=trials, iters=iters, slicing_repeat=slicing_repeat, # iters一般设成50，slicing_repeat一般设成4，trials是用的线程数，betas是运行的逆温度
        betas=betas, alpha = alpha               # 峰值性能/带宽 
    )
    print('Finish find contraction tree, next construct the scheme.')
    # tensors = []
    # for x in range(len(qc.tensors)):
    #     if x not in final_qubits:
    #         tensors.append(qc.tensors[x].to(device))

    scheme_sparsestate, _, bitstrings_sorted = contraction_scheme_sparse(
        ctree_new, bitstrings, sc_target=n)
    print('Finish construct the scheme.')
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

