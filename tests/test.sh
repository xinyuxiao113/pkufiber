mkdir data

python -m scripts.data

./scripts/train_ldbp.sh test configs/dsp/fdbp.yaml

./scripts/train_eq.sh test configs/0711/ampbcaddnn.yaml