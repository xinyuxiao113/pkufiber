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