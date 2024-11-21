import os
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)
from PyCmpltrtok.common import sep, parse_ids_list


def get_gpu_indexes_from_env(env_name='CUDA_VISIBLE_DEVICES'):
    gpus = os.environ.get(env_name, None)
    if gpus is None:
        return []
    else:
        indexes = parse_ids_list(gpus)
        return indexes


def print_gpu_utilization(idx):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(idx)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"#{idx} GPU memory occupied: {info.used//1024**2} MB.")
    
    
if '__main__' == __name__:
    sep('default')
    ids = get_gpu_indexes_from_env()
    print(ids)
    sep('CUDA_VISIBLE_DEVICES')
    ids = get_gpu_indexes_from_env('CUDA_VISIBLE_DEVICES')
    print(ids)
    sep('CPUS')
    ids = get_gpu_indexes_from_env('GPUS')
    print(ids)
    for xid in ids:
        print_gpu_utilization(xid)
