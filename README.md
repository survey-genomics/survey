# Survey

<p align="center">
    <picture align="center">
    <img src="https://github.com/survey-genomics/survey/blob/main/assets/logo_shine%405x.png" width="100px">
    </picture>
</p>
<p align="center">
</p>


The **Survey** toolkit is designed to process and analyze spatially-hashed single-cell data produced by the Survey Genomics assay. The core object is a MuData object ([docs](https://github.com/scverse/mudata), [repo](https://mudata.readthedocs.io/en/latest/#)) which stores single-cell data, spatial hash data, and any other modalities assayed concurrently (e.g. surface protein expression).


## System Requirements
Survey has been tested using Python 3.10 on MacOS and Ubuntu Linux.

## Installation

1. [Install mamba (or the latest conda).](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

2. Clone this repository and create a conda environment from the `environment.yml` file:

    ```
    cd survey/
    mamba env create -f environment.yml
    mamba activate survey-env
    ```

3. Then, install the package using pip:

    ```
    (survey-env) pip install .
    ```

## Environment File

The environment file has been configured for analysis in Jupyter Lab, tested within VS Code (Version: 1.103.1, Universal) using the extensions for Jupyter (version 2025.7.0) and Jupyter Powertoys (version 0.1.1). To recreate it, the following installations (beyond the dependencies listed in `pyproject.toml`) were performed, in order (all within survey-env):

```
mamba install bioconda::harmonypy==0.0.10 # harmony integration/batch correction
mamba install -c conda-forge python-igraph==0.11.5 # clustering
mamba install ipykernel=6.29.5 # Jupyter Lab
mamba install -c conda-forge ipywidgets # for tqdm and other notebook-rendered tools
mamba install tqdm # loading bar for long processes
mamba install -c conda-forge ipympl # in-notebook segmentation using mpl interactive
```

To use the environment as a python kernel for Jupyter Lab:

```
python -m ipykernel install --user --name survey-env --display-name "survey-env"
```


## Support

Please report issues or feature requests via GitHub.
