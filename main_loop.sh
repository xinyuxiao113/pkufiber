#!/bin/bash

M_values=(41)
rho_values=(1)
config=configs/0801/frepbc.yaml

for M_value in "${M_values[@]}"; do
    for rho_value in "${rho_values[@]}"; do
        overlaps=$(echo "$M_value - 1" | bc)
        strides=$(echo "$overlaps*4+1" | bc)
        echo "Running experiment with M=$M_value, rho=$rho_value, overlaps=$overlaps, strides=$strides"

        index=80G_3ch_frepbc_M"$M_value"_rho"$rho_value"_ol"$overlaps"_strides"$strides"

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
    done
done
