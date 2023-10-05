Installation
============
To install the Python implementation of tradeSeq in Python 3.X, run::

    conda create -n tradeSeq-py3X python=3.X --yes && conda activate tradeSeq-py3X
    git clone https://github.com/WeilerP/tradeSeq-py.git
    cd tradeSeq-py
    pip install -e .

To install the developer installation, run::

    pip install -e ".[dev]"
    pre-commit install
