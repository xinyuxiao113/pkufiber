import yaml 
import torch 
import re
import os
import numpy as np
import argparse
from tensorboard.backend.event_processing import event_accumulator
import pkufiber.dsp.nonlinear_compensation as nl
import pkufiber as pf


def load_model(model_name, model_info):
    model = getattr(nl, model_name)(**model_info)
    return model

def load_experiment(path):
    # 初始化 event accumulator
    logdir = path + '/logs/tensorboard'
    data = {}

    # load model config
    with open(path + '/config.yaml') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    data['config'] = config

    # load model 
    name = config['model_name'] if 'model_name' in config else config['dbp_name']
    info = config['model_info'] if 'model_info' in config else config['dbp_info']
    model = load_model(name, info)
    data['model']  = model

    # load history of training
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags()
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        epoch = []
        value = []
        for event in events:
            epoch.append(event.step)
            value.append(event.value)
        data[tag] = {"epoch": epoch, "value": value}

    # load qfactor-power 
    if os.path.exists(path + '/results/qfactor.npy'):
        data['qfactor_power'] = (np.load(path + '/results/qfactor.npy'), np.load(path + '/results/power.npy'))
        with open(path + '/results/test.yaml') as f: config = yaml.load(f, Loader=yaml.FullLoader)
        data['qfactor_power_config'] = config


    return data


def load_baseline(config_path, method='CDC'):
    '''
    method: CDC, DBP%d, %d = [1,2,4,8,16,32,64]
    '''
    with open(config_path) as f: config = yaml.load(f, Loader=yaml.FullLoader)

    metrics = []
    for pch in config['Pch']:
        data = pf.data.FiberDataset(path=config['path'], Nmodes=config['Nmodes'], 
                                    Nch=config['Nch'], Rs=config['Rs'], Pch=pch, 
                                    window_size=config['num_symb'], strides=1,  
                                    num_symb=config['num_symb'], truncate=config['truncate'], 
                                    Tx_window=True, pre_transform=f'Rx_{method}DDLMS(taps=32,lr=[0.015625, 0.0078125])')
        Rx,Tx,info = data[0]
        metrics.append(pf.qfactor_all(Rx, Tx))
    
    return metrics, config['Pch']
        