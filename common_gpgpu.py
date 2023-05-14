import os
from PyCmpltrtok.common import sep, parse_ids_list


def get_gpu_indexes_from_env(env_name='CUDA_VISIBLE_DEVICES'):
    gpus = os.environ.get(env_name, None)
    if gpus is None:
        return []
    else:
        indexes = parse_ids_list(gpus)
        return indexes


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
