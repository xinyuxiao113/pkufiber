import yaml
import argparse

# 读取 YAML 文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 修改参数
config['train_data']['Rs_list'] = [40]
config['train_data']['Nch_list'] = [1]
config['train_data']['Pch_list'] = [-2]

config['test_data']['Rs_list'] = [40]
config['test_data']['Nch_list'] = [1]
config['test_data']['Pch_list'] = [-2]

# 保存修改后的 YAML 文件
with open('config_modified.yaml', 'w') as file:
    yaml.safe_dump(config, file)

print("YAML file updated and saved as 'config_modified.yaml'.")
