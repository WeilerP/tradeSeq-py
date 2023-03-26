# tradeSeq - TRAjectory Differential Expression analysis for SEQuencing data

## General

This repository implements the R package [tradeSeq](https://www.bioconductor.org/packages/release/bioc/html/tradeSeq.html)
in Python. The code base is still experiencing heavy development. While we reimplement the tradeSeq package, exact
reproducibility is not guaranteed. Differences in result are caused by relying on Python packages instead of their R counterparts in some cases.

## Installation

To install the Python implementation of tradeSeq in Python 3.X, run

```bash
conda create -n tradeSeq-py3X python=3.X --yes && conda activate tradeSeq-py3X

git clone https://github.com/WeilerP/tradeSeq-py.git
cd tradeSeq-py

pip install -e .
```

To install the developer installation, run

```bash
pip install -e ".[dev]"
pre-commit install
```
