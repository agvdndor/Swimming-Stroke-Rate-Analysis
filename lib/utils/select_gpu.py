import os
import sys
import numpy as np
import torch

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    id = np.argmax(memory_available)
    mem = np.max(memory_available)
    return id, mem

def select_best_gpu(min_mem=7000):
    id, mem = get_free_gpu()

    if mem > min_mem:
        return torch.device('cuda:{}'.format(id))
    else:
        print("No GPU with enough resources available, available memory: {}".format(mem))
        sys.exit(0)

if __name__ == "__main__":
    id, mem = get_free_gpu()
    print('gpu id: {}, with memory: {} MiB'.format(id, mem))
