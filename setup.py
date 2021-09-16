import os
import versioneer
from setuptools import setup, find_packages

here = os.path.dirname(__file__)
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = ["xarray>=0.14.1", "dask", "numpy", "scipy"]
doc_requires = [
    "sphinx",
    "sphinxcontrib-srclinks",
    "sphinx-pangeo-theme",
    "numpydoc",
    "IPython",
    "nbsphinx",
]

extras_require = {
    "complete": install_requires,
    "docs": doc_requires,
}
extras_require["dev"] = extras_require["complete"] + [
    "pytest",
    "pytest-cov",
    "flake8",
    "black",
    "codecov",
]

setup(
    name="xarrayutils",
    description="A collection of various tools for data analysis built on top of xarray and xgcm",
    url="https://github.com/jbusecke/xarrayutils",
    author="xarrayutils Developers",
    author_email="julius@ldeo.columbia.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=long_description,
    long_description_content_type="text/markdown",
)
