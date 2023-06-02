import torch
import numpy as np
from copy import deepcopy
from os.path import exists, dirname, abspath
from os import makedirs
import multiprocessing as mp
from math import ceil
import time

def tensor_contraction_sparse(tensors, contraction_scheme, use_cutensor=False):
    '''
    contraction the tensor network according to contraction scheme

    :param tensors: numerical tensors of the tensor network
    :param contraction_scheme: 
        list of contraction step, defintion of entries in each step:
        step[0]: locations of tensors to be contracted
        step[1]: einsum equation of this tensor contraction
        step[2]: batch dimension of the contraction
        step[3]: optional, if the second tensor has batch dimension, 
            then here is the reshape sequence
        step[4]: optional, if the second tensor has batch dimension, 
            then here is the correct reshape sequence for validation

    :return tensors[i]: the final resulting amplitudes
    '''

    if use_cutensor:
        from cutensor.torch import EinsumGeneral
        einsum_func = EinsumGeneral
    else:
        einsum_func = torch.einsum

    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]
        if len(batch_i) > 1:
            tensors[i] = [tensors[i]]
            for k in range(len(batch_i)-1, -1, -1):
                if k != 0:
                    if step[3]:
                        tensors[i].insert(
                            1, 
                            einsum_func(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                            ).reshape(step[3])
                        )
                    else:
                        tensors[i].insert(
                            1, 
                            einsum_func(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]])
                        )
                else:
                    if step[3]:
                        tensors[i][0] = einsum_func(
                            step[1],
                            tensors[i][0][batch_i[k]], 
                            tensors[j][batch_j[k]], 
                        ).reshape(step[3])
                    else:
                        tensors[i][0] = einsum_func(
                            step[1],
                            tensors[i][0][batch_i[k]], 
                            tensors[j][batch_j[k]], 
                        )
            tensors[j] = []
            tensors[i] = torch.cat(tensors[i], dim=0)
        elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
            tensors[i] = tensors[i][batch_i[0]]
            tensors[j] = tensors[j][batch_j[0]]
            tensors[i] = einsum_func(step[1], tensors[i], tensors[j])
        elif len(step) > 3:
            tensors[i] = einsum_func(
                step[1],
                tensors[i],
                tensors[j],
            ).reshape(step[3])
            if len(batch_i) == 1:
                tensors[i] = tensors[i][batch_i[0]]
            tensors[j] = []
        else:
            tensors[i] = einsum_func(step[1], tensors[i], tensors[j])
            tensors[j] = []

    return tensors[i]


def contraction_single_task(
        tensors:list, scheme:list, slicing_indices:dict, 
        task_id:int, device='cuda:0', n_sub_task = 1, use_cutensor = False
    ):
    # n_sub_task: number of subtask of each task
    store_path = abspath(dirname(__file__)) + '/results/'
    if not exists(store_path):
        try:
            makedirs(store_path)
        except:
            pass
    file_path = store_path + f'partial_contraction_results_{task_id}.pt'
    time_path = store_path + f'result_time.txt'
    if not exists(file_path):
        t0 = time.perf_counter()
        slicing_edges = list(slicing_indices.keys())
        tensors_gpu = [tensor.to(device) for tensor in tensors]
        for s in range(task_id * n_sub_task, (task_id + 1) * n_sub_task):
            configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
            sliced_tensors = tensors_gpu.copy()
            for x in range(len(slicing_edges)):
                m, n = slicing_edges[x]
                idxm_n, idxn_m = slicing_indices[(m, n)]
                sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
                sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()
            if s == task_id * n_sub_task:
                collect_tensor = tensor_contraction_sparse(sliced_tensors, scheme, use_cutensor=use_cutensor)
            else:
                collect_tensor += tensor_contraction_sparse(sliced_tensors, scheme, use_cutensor=use_cutensor)
        t1 = time.perf_counter()
        torch.save(collect_tensor.cpu(), file_path)
        with open(time_path, 'a') as f:
            f.write(f'task id {task_id} running time: {t1-t0:.4f} seconds\n')
        print(f'subtask {task_id} done, the partial result file has been written into results/partial_contraction_results_{task_id}.pt')
    else:
        print(f'subtask {task_id} has already been calculated, skip to another one.')


def collect_results(task_num):
    for task_id in range(task_num):
        file_path = abspath(dirname(__file__)) + f'/results/partial_contraction_results_{task_id}.pt'
        if task_id == 0:
            collect_result = torch.load(file_path)
        else:
            collect_result += torch.load(file_path)
    
    return collect_result


def write_result(bitstrings, results):
    n_qubit = 30
    amplitude_filename = abspath(dirname(__file__)) + f'/results/result_amplitudes.txt'
    xeb_filename = abspath(dirname(__file__)) + f'/results/result_xeb.txt'
    time_filename = abspath(dirname(__file__)) + f'/results/result_time.txt'
    with open(amplitude_filename, 'w') as f:
        for bitstring, amplitude in zip(bitstrings, results):
            f.write(f'{bitstring} {np.real(amplitude)} {np.imag(amplitude)}j\n')
    # with open(xeb_filename, 'w') as f:
    #     f.write(f'{results.abs().square().mean().item() * 2 ** n_qubit - 1:.4f}')
    with open(time_filename, 'r') as f:
        lines = f.readlines()
    time_all = sum([float(line.split()[5]) for line in lines])
    with open(time_filename, 'a') as f:
        f.write(f'overall running time: {time_all:.2f} seconds.\n')


