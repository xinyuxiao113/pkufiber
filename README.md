# Installation 
pip install -e .


# 如果希望安装jax GPU版本和torch协同使用，应该首先手动安装
pip install torch & pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# 然后运行
pip install -e .


# 增加新模型的开发步骤
- 在dsp/nonlinear_compensation/下开发模型
- 在nonlinear_compensation/__init__.py 中注册模型
- 在scripts下train_eq.py, test_eq.py中增加设置关于Tx_window, window_size


# 生成数据
python -m scripts.data --config configs/data/base_test.yaml --path data/test.h5

# 训练模型
index=0926_eqfdbp_v3
config=configs/0926_dbp_sps1/fdbp.yaml
./scripts/train_eq.sh $index $config

# 测试模型
python -m scripts.test_eq --path experiments/$index --test_config configs/dsp/test_eq.yaml
