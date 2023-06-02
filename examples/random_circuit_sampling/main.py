import argparse
from os.path import abspath, dirname, exists

import numpy as np
import torch
from network_contraction import collect_results, contraction_single_task, write_result
from search_order import read_samples, search_order

# 生成缩并顺序，这一步需要用artensor库。可以使用已经生成的缩并顺序跳过这一步
need_search_order = True
if need_search_order:
    search_order(
        n=30,
        m=14,
        seq="EFGH",
        device="cuda",
        sc_target=30,
        seed=0,
        bitstrings_txt="amplitudes_n30_m14_s0_e0_pEFGH_10000.txt",
        max_bitstrings=1_000,
    )

contraction_filename = abspath(dirname(__file__)) + "/scheme_n30_m14.pt"
n_sub_task = 1
task_num = 1
max_bitstrings = 1_000
use_cutensor = False  # 使用cutenor可以获得更高性能

if not exists(contraction_filename):
    assert ValueError("No contraction data!")
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-taskid", type=int, default=0, help="tensor network contraction id"
)
parser.add_argument(
    "-device",
    type=int,
    default=0,
    help="using which device, -1 for cpu, otherwise use cuda:device",
)
args = parser.parse_args()
assert args.taskid >= 0 and args.taskid <= 15
assert args.device >= -1
args.device = "cuda:0"

"""
There will be four objects in the contraction scheme:
    tensors: Numerical tensors in the tensor network
    scheme: Contraction scheme to guide the contraction of the tensor network
    slicing_indices: Indices to be sliced, the whole tensor network will be
        divided into 2**(num_slicing_indices) sub-pieces and the contraction of
        all of them returns the overall result. The indices is sliced to avoid
        large intermediate tensors during the contraction.
    bitstrings: bitstrings of interest, the contraction result will be amplitudes
        of these bitstrings
"""
tensors, scheme, slicing_indices, bitstrings = torch.load(contraction_filename)
contraction_single_task(
    tensors,
    scheme,
    slicing_indices,
    args.taskid,
    args.device,
    n_sub_task=n_sub_task,
    use_cutensor=use_cutensor,
)

file_exist_flag = True
for i in range(task_num):
    if not exists(
        abspath(dirname(__file__)) + f"/results/partial_contraction_results_{i}.pt"
    ):
        file_exist_flag = False
if file_exist_flag:
    print("collecting results, results will be written into results/result_*.txt")
    results = collect_results(task_num)
    write_result(bitstrings, results)

    # 将计算结果与Google的数据对比
    bitstrings_sorted = bitstrings
    amplitude_sparsestate = results
    correct_num = 0
    data = read_samples("amplitudes_n30_m14_s0_e0_pEFGH_10000.txt")
    amplitude_google = np.array([data[i][1] for i in range(max_bitstrings)])
    bitstrings = [data[i][0] for i in range(max_bitstrings)]
    for i in range(len(bitstrings_sorted)):
        ind_google = bitstrings.index(bitstrings_sorted[i])
        relative_error = abs(
            amplitude_sparsestate[i].item() - amplitude_google[ind_google]
        ) / abs(amplitude_google[ind_google])
        if relative_error <= 0.05:
            correct_num += 1
    print(f"bitstring amplitude correct ratio:{correct_num/max_bitstrings}")
