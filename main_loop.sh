#!/bin/bash

ntaps_values=(21 41 81)
rho_values=(0.02 0.04 0.08)
ol_values=(10 20 40 80)
strides=(11 21 41 81)
config=configs/0909/fredbp.yaml

for ntaps_value in "${ntaps_values[@]}"; do
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
