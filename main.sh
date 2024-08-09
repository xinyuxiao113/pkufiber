#!/bin/bash

M_value=201
rho_value=1
overlaps=100
strides=401
config=configs/0801/frepbc.yaml
echo "Running experiment with M=$M_value, rho=$rho_value, overlaps=$overlaps, strides=$strides"
index=80G_3ch_frepbc_M"$M_value"_rho"$rho_value"_ol"$overlaps"_strides"$strides"

# modify the yaml file
python -m scripts.modify_yaml $config $config train_data.num_symb_per_mode 1600000
python -m scripts.modify_yaml $config $config batch_size 200


# modify the yaml file
python -m scripts.modify_yaml $config $config model_info.M $M_value
python -m scripts.modify_yaml $config $config model_info.rho $rho_value
python -m scripts.modify_yaml $config $config model_info.overlaps $overlaps
python -m scripts.modify_yaml $config $config model_info.strides $strides
python -m scripts.modify_yaml $config $config train_data.strides $strides
python -m scripts.modify_yaml $config $config test_data.strides $strides


# training
./scripts/train_eq.sh $index $config

# testing
python -m scripts.test_eq --path experiments/$index --test_config configs/dsp/test_eq.yaml