
start=1
end=3
step=0.1

value=$start
while (( $(echo "$value <= $end" | bc -l) )); do
    echo RKN:p=$value
    value=$(echo "$value + $step" | bc)

    # define config file
    python -m scripts.modify_yaml configs/0801/frepbc_rkn.yaml configs/0801/frepbc_rkn.yaml p  $value

    # run the experiment
    ./scripts/train_eq.sh 80G_3ch_frepbc_M41_rho1_ol40_strides41_p$value configs/0801/frepbc_rkn.yaml
done