from pathlib import Path

from setuptools import setup, find_packages

try:
    from tradeseq import __email__, __author__, __version__, __maintainer__
except ImportError:
    __author__ = ""
    __maintainer__ = "Michal Klein"
    __email__ = ""
    __version__ = "0.0.0"

# TODO(michalk8): use only toml file?
setup(
    name="tradeseq",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    description=Path("README.rst").read_text("utf-8").splitlines()[2],
    long_description=Path("README.rst").read_text("utf-8"),
    long_description_content_type="text/x-rst; charset=UTF-8",
    url="https://github.com/michalk8/tradeSeq-py",  # TODO(michalk8): change me
    download_url="https://pypi.org/michalk8/tradeseq",  # TODO(michalk8): change me
    project_urls={
        "Documentation": "https://tradeseq.readthedocs.io/en/stable",  # TODO(michalk8): change me
        "Source Code": "https://github.com/michalk8/tradeSeq-py",  # TODO(michalk8): change me
    },
    license="MIT",
    platforms=["Linux", "MacOSX"],
    packages=find_packages(),
    zip_safe=False,
    install_requires=Path("requirements.txt").read_text("utf-8").splitlines(),
    # TODO(michalk8): add test and doc requirements
    extras_require={
        "dev": ["pre-commit>=2.16.0", "pytest>=6.2.2", "pytest-cov"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",  # TODO(michalk8): change me
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Typing :: Typed",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    # TODO(michalk8): populate
    keywords=sorted(["single-cell", "bio-informatics", "lineage", "differential expression"]),
)
