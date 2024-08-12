#!/bin/bash

M_values=(81 161)
rho_values=(1 2 4 8)
config=configs/0801/ampbcstep.yaml

for M_value in "${M_values[@]}"; do
    for rho_value in "${rho_values[@]}"; do
        echo "Running experiment with M=$M_value, rho=$rho_value"
        index=80G_3ch_ampbcstep_M"$M_value"_rho"$rho_value"
        # modify the yaml file
        python -m scripts.modify_yaml $config $config model_info.M $M_value
        python -m scripts.modify_yaml $config $config model_info.rho $rho_value

        # training
        ./scripts/train_eq.sh $index $config

        # testing
        python -m scripts.test_eq --path experiments/$index --test_config configs/dsp/test_eq.yaml
    done
done
