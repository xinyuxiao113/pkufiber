
index=80G_3ch_pbc_M41_rho2

./scripts/train_eq.sh $index configs/0801/pbcstep.yaml

python -m scripts.test_eq --path experiments/$index --test_config configs/dsp/test_eq.yaml