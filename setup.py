from setuptools import setup, find_packages

setup(
    name="fedlab-zmq",
    version="1.0.0",
    description="Federated Learning Research Framework — ZeroMQ transport (UM6P)",
    author="J. Nikiema & E. Amhoud",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyzmq>=25.1.0",
        "msgpack>=1.0.8",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "streamlit>=1.35.0",
        "plotly>=5.20.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": ["fedlab=cli.fedlab:main"],
    },
)
