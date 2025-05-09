'''
1. show Q factor - power curve. 
2. show constellation.
'''

import pkufiber as pf
import yaml 
import os
import re
import argparse
import torch 
import numpy as np
from scripts.train_ldbp import init_model, DataLoader, test_model


def find_max_number(file_list):
    # 使用正则表达式提取数字部分
    numbers = [int(re.search(r'(\d+)\.pth$', file).group(1)) for file in file_list] # type: ignore
    # 找到最大值
    max_number = max(numbers)
    return max_number


def load_latest_model(path):
    models = os.listdir(path + '/models')
    n = find_max_number(models)
    with open(path + '/config.yaml') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    model = init_model(config['dbp_name'], config['dbp_info'])
    param_dict = torch.load(path + f'/models/{n}.pth', map_location='cpu')['dbp_param']
    model.load_state_dict(param_dict) 
    print(f'model{n}.pth loaded.')
    return model


def test_all(model_index: str, test_config: str):

    with open(model_index + '/config.yaml') as f: model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = yaml.load(open(test_config), Loader=yaml.FullLoader)
    model = load_latest_model(model_index)
    model = model.to(data_cfg['device'])

    strides = data_cfg['strides']
    Tx_window = model_cfg['test_data']['Tx_window'] 
    window_size = model.overlaps + strides + (data_cfg['taps'] - 1)//2

    print(f"strides = {strides}, window_size = {window_size}, Tx_window = {Tx_window}")

    Q = {}

    for pch in data_cfg['Pch']:
        for rs in data_cfg['Rs']:
            for nch in data_cfg['Nch']:
                print(f'Testing power {pch} dBm ...{rs} G, {nch} channels')
                data = pf.data.FiberDataset(path=data_cfg['path'], Nmodes=data_cfg['Nmodes'], Nch=nch, Rs=rs, Pch=pch, 
                                window_size=window_size, strides=strides,  num_symb=data_cfg['num_symb'], truncate=data_cfg['truncate'], Tx_window=Tx_window, pre_transform=data_cfg['pre_transform'])
                dataloader = DataLoader(data, batch_size=data_cfg['batch_size'], shuffle=False, drop_last=False)
                print(f'Test data size: {len(data)}, batch size: {data_cfg["batch_size"]}, batch number: {len(dataloader)}')
                metric, (y,x) = test_model(model, dataloader, device=data_cfg['device'])
                Q[f'power {pch} dBm, {rs} G, {nch} channel'] = metric['Qsq']

    return Q


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='path to the model')
    parser.add_argument('--test_config', type=str, default=None, help='path to the test config yaml.')
    args = parser.parse_args()

    Q = test_all(args.path, args.test_config)
    torch.save(Q, args.path + '/Q.pth')
