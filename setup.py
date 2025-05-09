
from setuptools import setup, find_packages

setup(
    name='pkufiber',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'h5py',
        'pyyaml',
        'scikit-commpy',
        'tqdm',
        'tensorboard',
        'torch',
        'jupyter',
        'neuraloperator',
        'seaborn',
        'jax',
        'flax',
    ],
    python_requires='>=3.7, <4',
    entry_points={
        'console_scripts': [
            'pkufiber_simulation = pkufiber.data.generator:main',
            'pkufiber_compensation = pkufiber.data.compensation:main',
            'pkufiber_train_ldbp = scripts.train_ldbp:main',
            'pkufiber_train_eq = scripts.train_eq:main',
        ]
    }
)
