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
from scripts.train_eq import init_model, DataLoader, test_model



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
    model = init_model(config['model_name'], config['model_info'])
    param_dict = torch.load(path + f'/models/{n}.pth', map_location='cpu')['model_param']
    model.load_state_dict(param_dict) 
    print(f'model{n}.pth loaded.')
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='path to the model')
    parser.add_argument('--test_config', type=str, default=None, help='path to the test config yaml.')
    args = parser.parse_args()


    with open(args.path + '/config.yaml') as f: model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = yaml.load(open(args.test_config), Loader=yaml.FullLoader)
    model = load_latest_model(args.path)
    model = model.to(data_cfg['device'])

    strides = model_cfg['test_data']['strides'] 
    Tx_window = True if model_cfg['model_name'] in ['MultiStepAMPBC', 'MultiStepPBC', 'EqFno', 'EqFrePBC', 'EqAMPBCstep', 'EqPBCstep'] else False
    window_size = model.overlaps + strides

    print(f"strides = {strides}, window_size = {window_size}, Tx_window = {Tx_window}")

    Q = []
    if not os.path.exists(args.path + '/results'):
        os.makedirs(args.path + '/results')


    for pch in data_cfg['Pch']:
        print(f'Testing power {pch} dBm ...')
        data = pf.data.FiberDataset(path=data_cfg['path'], Nmodes=data_cfg['Nmodes'], Nch=data_cfg['Nch'], Rs=data_cfg['Rs'], Pch=pch, 
                        window_size=window_size, strides=strides,  num_symb=data_cfg['num_symb'], truncate=data_cfg['truncate'], Tx_window=Tx_window, pre_transform=data_cfg['pre_transform'])
        dataloader = DataLoader(data, batch_size=data_cfg['batch_size'], shuffle=False, drop_last=False)
        print(f'Test data size: {len(data)}, batch size: {data_cfg["batch_size"]}, batch number: {len(dataloader)}')
        metric, (y,x) = test_model(model, dataloader, device=data_cfg['device'])
        np.save(args.path + f'/results/y_{pch}dBm.npy', y)
        np.save(args.path + f'/results/x_{pch}dBm.npy', x)
        Q.append(metric['Qsq'])

    print('Result are stored in:', args.path + '/results')
    os.system(f'cp {args.test_config} {args.path}/results/test.yaml')
    np.save(args.path + '/results/qfactor.npy', Q) 
    np.save(args.path + '/results/power.npy', data_cfg['Pch'])

    # plot Q factor - power curve


    # plot constellation


    # plot eye diagram


