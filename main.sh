# 遍历configs/paper_config所有文件
for file in $(ls configs/paper_config/1023)
do
    ./scripts/train_eq.sh 1023/$file  configs/paper_config/1023/$file/config.yaml
    python -m scripts.test_eq --path experiments/1023/$file --test_config configs/test_eq.yaml
done

for file in $(ls configs/paper_config/1024)
do
    ./scripts/train_eq.sh 1024/$file  configs/paper_config/1024/$file/config.yaml
    python -m scripts.test_eq --path experiments/1024/$file --test_config configs/test_eq.yaml
done


for file in $(ls configs/paper_config/1025)
do
    ./scripts/train_eq.sh 1025/$file  configs/paper_config/1025/$file/config.yaml
    python -m scripts.test_eq --path experiments/1025/$file --test_config configs/test_eq.yaml
done


for file in $(ls configs/paper_config/1105)
do
    ./scripts/train_eq.sh 1105/$file  configs/paper_config/1105/$file/config.yaml
    python -m scripts.test_eq --path experiments/1105/$file --test_config configs/test_eq.yaml
done



./scripts/train_ldbp.sh 80G_3ch_fdbp_v1 configs/paper_config/80G_3ch_fdbp_v1/config.yaml
python -m scripts.test_ldbp --path experiments/80G_3ch_fdbp_v1 --test_config configs/test_ldbp.yaml

./scripts/train_ldbp.sh 80G_3ch_fdbp_v7 configs/paper_config/80G_3ch_fdbp_v7/config.yaml
python -m scripts.test_ldbp --path experiments/80G_3ch_fdbp_v7 --test_config configs/test_ldbp.yaml