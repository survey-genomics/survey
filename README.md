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

    ```zsh
    cd survey/
    mamba env create -f environment_macos.yml
    mamba activate survey-env
    ```

3. Then, install the package using pip:

    ```zsh
    (survey-env) pip install .
    ```

## Environment File

The environment files have been configured for analysis in Jupyter Lab, tested within VS Code (Version: 1.103.1, Universal) using the extensions for Jupyter (version 2025.7.0) and Jupyter Powertoys (version 0.1.1). If they cannot be solved on your machine, you should be able to recreate them with the following:

```zsh
mamba create -n survey-env python=3.10.18
mamba activate survey-env
git clone <this_repo>
cd survey
pip install .
mamba install bioconda::harmonypy==0.0.10
mamba install -c conda-forge python-igraph==0.11.5
mamba install ipykernel=6.29.5
mamba install -c conda-forge ipywidgets==8.1.7
mamba install tqdm==4.67.1
mamba install -c conda-forge ipympl==0.9.7
```

Explanations of the packages installed:
```txt
bioconda::harmonypy==0.0.10 # harmony integration/batch correction
python-igraph==0.11.5 # clustering
ipykernel=6.29.5 # Jupyter Lab
ipywidgets==8.1.7 # for tqdm and other notebook-rendered tools
tqdm==4.67.1 # loading bar for long processes
ipympl==0.9.7 # in-notebook segmentation using mpl interactive
```

To use the environment as a python kernel for Jupyter Lab:

```
python -m ipykernel install --user --name survey-env --display-name "survey-env"
```

## Support

Please report issues or feature requests via GitHub.
