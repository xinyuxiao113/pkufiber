import pkufiber as pf
import matplotlib.pyplot as plt, numpy as np
from pkufiber.data.loader import FiberDataset
# path = '/home/xiaoxinyu/dataset/test.h5'
# path = '/home/xiaoxinyu/TorchFiber/dataset_A800/test.h5'
path = '/gpfs/share/home/2001110035/TorchFiber/dataset/test.h5'


def get_data(path, method='CDC'):
    q_list = []
    for pch in range(2, 3):
        data = FiberDataset(path, Nch=1, Rs=40, Pch=pch, Nmodes=2,
                        window_size=80000, strides=1, num_symb=80000, truncate=10000, 
                        Tx_window=True, pre_transform=f'Rx_{method}DDLMS(taps=32,lr=[0.015625, 0.0078125])')
        Rx,Tx, info = data[0]
        q_list.append(pf.qfactor_all(Rx, Tx))
    return q_list

q1 = get_data(path, 'CDC')
print(q1)