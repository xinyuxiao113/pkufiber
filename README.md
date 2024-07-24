# Installation 
pip install -e .


# 如果希望安装jax GPU版本和torch协同使用，应该首先手动安装
pip install torch & pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# 然后运行
pip install -e .