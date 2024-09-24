import pickle, torch, numpy as np, time, random, os, h5py
from torch.utils.data import Dataset, ConcatDataset

from pkufiber.core import TorchInput, TorchSignal, TorchTime
from pkufiber.simulation import ber

'''

train.h5 -- dataset

group:
    - pulse
    - SymbTx  
    - SignalRx
    - Rx(sps=2,chid=0,method=frequency cut)
        - info            # [Pch, Fi, Rs, Nch]
        - Tx
        - Rx
        - Rx_CDC 
        - Rx_DBP%d
        - Rx_CDCDDLMS
        - Rx_DBP%dDDLMS
        - ...
    - Rx(sps=2,chid=0,method=filtering)
        - ...
    """
'''


class FiberDataset(Dataset):
    """
    Dataset for fiber optic data.

    Attributes:
        path (str): Path to the dataset file.
        Nmodes (int): Number of modes.
        Nch (int): Number of channels.
        Rs (float): Symbol rate.
        Pch (float): Power per channel.
        window_size (int): Size of the window.
        strides (int): Stride length.
        truncate (int): Amount to truncate.
        Tx_window (bool): Whether to use Tx window.
        pre_transform (str): Pre-transformation method.

        window_size = strides + overlaps
        overlaps =  |start| + |stop|
    """

    def __init__(
        self,
        path: str = "dataset/test.h5",
        Nmodes: int = 2,
        Nch: int = 3,
        Rs: float = 40,
        Pch: float = -1,
        window_size: int = 41,
        strides: int = 1,
        num_symb: int = 10000000,
        truncate: int = 20000,
        Tx_window: bool = False,
        rx_grp: str = "Rx(sps=2,chid=0,method=frequency cut)",
        pre_transform: str = "Rx",
    ):
        assert (
            window_size > 0 and strides > 0
        ), "window_size and strides should be positive"
        assert num_symb >= window_size, "num_symb should be larger than window_size"
        self.path = path
        self.Nch = Nch
        self.Rs = Rs
        self.Pch = Pch
        self.Nmodes = Nmodes
        self.window_size = window_size
        self.pre_transform = pre_transform
        self.strides = strides
        self.truncate = truncate
        self.Tx_window = Tx_window

        self.Tx = []  # list of  [L, Nmodes]
        self.Rx = []  # list of  [L*sps, Nmodes]
        self.info = []  # list of  [4]
        self.grp_attrs = []

        if not os.path.exists(self.path):
            raise FileNotFoundError(f'{path} not exist!')
        
        with h5py.File(path, "r") as f:
            total_length = 0

            for key in f.keys():
                group = f[key]
                if (
                    group.attrs["Nch"] == self.Nch
                    and group.attrs["Rs(GHz)"] == self.Rs
                    and group.attrs["Nmodes"] == Nmodes
                    and group.attrs["Pch(dBm)"] == self.Pch
                ):
                    assert isinstance(group, h5py.Group)
                    subgrp = group[rx_grp]
                    assert isinstance(subgrp, h5py.Group)
                    if pre_transform not in subgrp.keys():
                        raise ValueError(
                            f"No such pre_transform in {rx_grp}, please choose from {subgrp.keys()}"
                        )
                    assert isinstance(subgrp, h5py.Group)

                    s = subgrp[pre_transform].attrs["start"]
                    e = subgrp[pre_transform].attrs["stop"] + subgrp[pre_transform].shape[1]  # type: ignore
                    sps = subgrp[pre_transform].attrs["sps"]

                    Tx_elements = subgrp["Tx"]
                    Rx_elements = subgrp[pre_transform]
                    info_elements = subgrp["info"]

                    for i in range(Tx_elements.shape[0]):  # type: ignore
                        tx_length = e - s - truncate

                        if total_length + tx_length > num_symb:
                            tx_length = num_symb - total_length

                        if total_length < num_symb:
                            self.Tx.append(torch.from_numpy(Tx_elements[i, truncate + s : truncate + s + tx_length]).to(torch.complex64))  # type: ignore
                            self.Rx.append(torch.from_numpy(Rx_elements[i, truncate * sps : (truncate + tx_length) * sps]).to(torch.complex64))  # type: ignore
                            self.info.append(torch.from_numpy(info_elements[i, ...]).to(torch.float32))  # type: ignore
                            self.grp_attrs.append(dict(group.attrs))
                            total_length += tx_length

                        if total_length >= num_symb:
                            break

                if total_length >= num_symb:
                    break

        if total_length < num_symb:
            print("Warning: not enough data, only {} symbols".format(total_length))

        if self.Tx == []:
            raise ValueError("No such dataset")

        self.Rx_sps = sps
        self.length_list = [
            (tx.shape[0] - self.window_size) // self.strides + 1 for tx in self.Tx
        ]
        self.length = sum(self.length_list)

    def get_info(self):
        '''
        Return the meta information about the first signal.
        '''
        assert len(self.grp_attrs) > 0
        print('number of signals: ', len(self.grp_attrs))
        return self.grp_attrs[0]

    def locate(self, idx):
        """
        return the index of batch i and the index of window j in the batch.
        """
        i = np.argmax(np.cumsum(self.length_list) > idx)
        j = int(idx - np.sum(self.length_list[:i]))
        return i, j

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Return Rx, Tx, info
        if Tx_window is True,
            shape: [window_size*sps, Nmodes], [windowsize, Nmodes], [4]
        else:
            shape: [window_size*sps, Nmodes], [Nmodes], [4]
            info: [Pch(dBm), Fi(Hz), Fs(Hz), Nch]   power per channel, carrier frequency, samplerate, number of channels
        """
        i, j = self.locate(idx)

        if self.Tx_window:
            return self.Rx[i][j * self.strides * self.Rx_sps : (j * self.strides + self.window_size) * self.Rx_sps, :], self.Tx[i][j * self.strides : j * self.strides + self.window_size, :], self.info[i]  # type: ignore  [M, Nmodes], [Nmodes]
        else:
            return self.Rx[i][j * self.strides * self.Rx_sps : (j * self.strides + self.window_size) * self.Rx_sps, :], self.Tx[i][j * self.strides + (self.window_size // 2), :], self.info[i]  # type: ignore


class MixFiberDataset(Dataset):
    """
    Dataset for mixing fiber optic data from multiple parameter configurations.

    Attributes:
        datasets (ConcatDataset): A concatenation of multiple FiberDataset objects.
    """

    def __init__(
        self,
        path,
        Nmodes,
        Nch_list,
        Rs_list,
        Pch_list,
        window_size=41,
        strides=1,
        num_symb_per_mode=10000000,
        truncate=20000,
        Tx_window=False,
        rx_grp="Rx(sps=2,chid=0,method=frequency cut)",
        pre_transform="Rx",
    ):
        datasets = []
        for Nch in Nch_list:
            for Rs in Rs_list:
                for Pch in Pch_list:
                    dataset = FiberDataset(
                        path=path,
                        Nmodes=Nmodes,
                        Nch=Nch,
                        Rs=Rs,
                        Pch=Pch,
                        window_size=window_size,
                        strides=strides,
                        num_symb=num_symb_per_mode,
                        truncate=truncate,
                        Tx_window=Tx_window,
                        rx_grp=rx_grp,
                        pre_transform=pre_transform,
                    )
                    datasets.append(dataset)

        self.datasets = ConcatDataset(datasets)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]



class PbcDataset(FiberDataset):

    def __init__(
        self,
        path: str = "dataset/test.h5",
        Nmodes: int = 2,
        Nch: int = 3,
        Rs: float = 40,
        Pch: float = -1,
        window_size: int = 41,
        num_symb: int = 10000000,
        truncate: int = 20000,
        ):

        super(PbcDataset, self).__init__(
            path=path,
            Nmodes=Nmodes,
            Nch=Nch,
            Rs=Rs,
            Pch=Pch,
            window_size=window_size,
            strides=1,
            num_symb=num_symb,
            truncate=truncate,
            Tx_window=False,
            pre_transform='Rx_CDCDDLMS(taps=32,lr=[0.015625, 0.0078125])',
        )


