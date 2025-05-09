#!/bin/bash

# 使用方法: ./scripts/train_eq.sh experiment_name config_file

EXPERIMENT_NAME=$1
CONFIG_FILE=$2

# 创建实验文件夹
exp_path=experiments/$EXPERIMENT_NAME
log_path=experiments/$EXPERIMENT_NAME/logs/tensorboard
model_path=experiments/$EXPERIMENT_NAME/models
mkdir -p $exp_path
mkdir -p $log_path
mkdir -p $model_path
cp $CONFIG_FILE $exp_path/config.yaml

# 启动训练脚本并保存日志
pkufiber_train_eq --config $CONFIG_FILE --log_path $log_path --model_path $model_path | tee experiments/$EXPERIMENT_NAME/logs/train.log

echo "Experiment $EXPERIMENT_NAME finished."

