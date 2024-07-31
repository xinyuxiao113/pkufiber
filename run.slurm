#!/bin/bash
#SBATCH -J xinyu-simulation            # Job名
#SBATCH -o run.out           # 输出, 目录./out必须存在，否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short        # qos(quality of service): normal or short or debug, 对应不同优先级及最大可用时长.
#SBATCH -p RTX3090          # 指定partition: geforce,k80,etc.
#SBATCH --cpus-per-task=8  # 申请 cpu core 数; 可用内存与申请 cpu core 数成正比.
#SBATCH --mem=40G          # 申请10G内存
#SBATCH --gres=gpu:1       # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 10:00:00        # 申请Job运行时长0小时5分钟0秒,若要申请超过一天时间,如申请1天,书写格式为#SBATCH -t 1-00:00:00

# 上述 SBATCH 参数不指定时均有系统指定的默认值


./scripts/train_eq.sh 80G_3ch_6db_sonn_v1  configs/dsp/sonn.yaml 

# ./scripts/train_eq.sh 80G_3ch_6db_ampbc_v3  configs/0723/ampbc.yaml 

# ./scripts/train_ldbp.sh pbcdbp configs/0717/pbcdbp.yaml
# python -m scripts.data --config configs/data/40G_3ch.yaml --path data/40G_3ch_batch40.h5
# ./scripts/train_eq.sh 80G_3ch_ampbc_v3_decision1_NFm10 configs/0717/ampbc.yaml
# ./scripts/train_eq.sh 80G_3ch_ampbcaddnn_v4_NFm10 configs/0717/ampbcaddnn.yaml
# ./scripts/train_eq.sh fno_v6 configs/0717/fno.yaml
# python -m scripts.data --config configs/data/80G_3ch.yaml --path data/80G_3ch_batch40_NFm10.h5


# ./scripts/train_eq.sh 80G_3ch_ampbc_ configs/0718/ampbc.yaml

# ./scripts/train_eq.sh 40G_3ch_ampbcaddnn_v4 configs/0718/ampbcaddnn.yaml

# ./scripts/train_eq.sh 40G_3ch_pbcnn_test_v1 configs/0718/pbcnn.yaml
# ./scripts/train_eq.sh 40G_3ch_sopbc_v3  configs/0718/sopbc.yaml 
# ./scripts/train_eq.sh 40G_3ch_multisteppbc_v3 configs/0718/multisteppbc.yaml

# ./scripts/train_eq.sh 80G_3ch_6db_ampbc_v3  configs/0723/ampbc.yaml 
# ./scripts/train_eq.sh 80G_3ch_6db_ampbcaddnn_v4  configs/0723/ampbcaddnn.yaml 
# ./scripts/train_eq.sh 80G_3ch_6db_sopbc_v1  configs/0723/sopbc.yaml 
# ./scripts/train_eq.sh 80G_3ch_6db_ampbcaddfno_v1  configs/0723/ampbcaddfno.yaml 
