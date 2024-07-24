import pkufiber as pf
import yaml 
import torch 
from scripts.train_eq import init_model, check_data_config, DataLoader, test_model

path = 'experiments/40G_3ch_ampbc_v1'
with open(path + '/config.yaml') as f: config = yaml.load(f, Loader=yaml.FullLoader)
param_dict = torch.load(path + '/models/60.pth', map_location='cpu')['model_param']

model = init_model(config['model_name'], config['model_info'])
model.load_state_dict(param_dict) 