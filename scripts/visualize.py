import yaml 
import torch 
import re
import os
import argparse
from tensorboard.backend.event_processing import event_accumulator
import pkufiber.dsp.nonlinear_compensation as nl


def load_model(model_name, model_info):
    model = getattr(nl, model_name)(**model_info)
    return model

def find_max_number(file_list):
    # 使用正则表达式提取数字部分
    numbers = [int(re.search(r'(\d+)\.pth$', file).group(1)) for file in file_list] # type: ignore
    # 找到最大值
    max_number = max(numbers)
    return max_number

def read_tensorboard_log(path):
    # 初始化 event accumulator
    logdir = path + '/logs/tensorboard'

    # history
    ea = event_accumulator.EventAccumulator(logdir)
    with open(path + '/config.yaml') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    model = load_model(config['model_name'], config['model_info'])
    # models = os.listdir(path + '/models')
    # n = find_max_number(models)
    # dic = torch.load(path + f'/models/{n}.pth')
    # model.load_state_dict(dic['model']) 

    ea.Reload()
    tags = ea.Tags()
    data = {}
    data['config'] = config
    data['model']  = model
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        epoch = []
        value = []
        for event in events:
            epoch.append(event.step)
            value.append(event.value)

        data[tag] = {"epoch": epoch, "value": value}
    
    return data

log_directory = "experiments/80G_3ch_frepbc_M41_rho1_ol40_strides41_p1.8"
data = read_tensorboard_log(log_directory)