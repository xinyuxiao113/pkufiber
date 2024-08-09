'''
Generate data for training and testing.

Usage:
    python -m scripts.data --config configs/data/base_test.yaml --path data/test.h5
'''
import pkufiber as pf 
import argparse, yaml

from pkufiber.data import fiber_simulation, data_compensation

parser = argparse.ArgumentParser(description='Generate data for training')  
parser.add_argument('--config', type=str, default='configs/data/base_test.yaml', help='path to config file')
parser.add_argument('--path', type=str, default='data/test.h5', help='path to save data')
args = parser.parse_args()
with open(args.config, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)

if type(config['tx']['Pch_dBm']) == list:
    Pch_dBm = config['tx']['Pch_dBm']
else:
    Pch_dBm = [config['tx']['Pch_dBm']]


for pch in Pch_dBm:
    config['tx']['Pch_dBm'] = pch 
    fiber_simulation(args.path, config)


data_compensation(args.path, 'CDC', device=config['rx']['device'])
data_compensation(args.path, 'DBP', stps=1, device=config['rx']['device'])
data_compensation(args.path, 'DBP', stps=2, device=config['rx']['device'])
data_compensation(args.path, 'DBP', stps=4, device=config['rx']['device'])
data_compensation(args.path, 'DBP', stps=8, device=config['rx']['device'])
data_compensation(args.path, 'DBP', stps=16, device=config['rx']['device'])
