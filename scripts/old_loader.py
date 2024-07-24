import pickle, h5py
import torch
from torch.utils.data import Dataset

train_data = pickle.load(open('/home/xiaoxinyu/TorchFiber/data/Nmodes2_large/Nch3_Rs40_Pch2_batch40_seed2671.pkl','rb'))
# train_data = pickle.load(open('/home/xiaoxinyu/TorchFiber/data/Nmodes2/train_batch10_4e5_afterCDCDSP.pkl','rb'))
test_data = pickle.load(open('/home/xiaoxinyu/TorchFiber/data/Nmodes2/test_batch10_4e5_afterCDCDSP.pkl','rb'))


def get_k(Nch, Rs, P, train_t, sps=2) -> list[int]:
    '''
        Return index of batch in train_t infomation with number of channels = Nch, symbol rate = Rs, Power=P.
            train_t: [B, 4].  [P, Fi, Fs, Nch]
    '''
    dis = torch.mean(torch.abs(train_t[:,[0,2,3]] - torch.tensor([P, Rs*sps*1e9,  Nch])), dim=1)
    k = torch.where(dis < 0.1)[0]
    if len(k) == 0:
        print('No matched data')
        raise ValueError
    else:
        # print('match batch: ', k)
        return k.tolist()




class OldData(Dataset):
    
    def __init__(self,mode='train', window_size=41, num_symb=400000, truncate=10000):
        self.window_size = window_size
        self.truncate = truncate
        if mode == 'train':
            self.Rx, self.Tx, self.info = train_data
            # k = get_k(3, 40, 2, train_data[-1])
            # self.Rx, self.Tx, self.info = train_data[0][k], train_data[1][k], train_data[2][k]
        else:
            k = get_k(3, 40, 2, test_data[-1])
            self.Rx, self.Tx, self.info = test_data[0][k], test_data[1][k], test_data[2][k]

        self.batch = self.Rx.shape[0]
        self.L = self.Rx.shape[1]
        self.num_symb = num_symb

    def __len__(self):
        return min((self.L - self.window_size + 1 - self.truncate) * self.batch, self.num_symb)
    
    def __getitem__(self, index):
        i,j = index//(self.L - self.window_size + 1 - self.truncate), index % (self.L - self.window_size + 1 - self.truncate)
        return self.Rx[i, self.truncate + j:self.truncate + j+self.window_size], self.Tx[i, self.truncate + j+(self.window_size//2)], self.info[i]
