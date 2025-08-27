# Survey

<p align="center">
    <picture align="center">
    <img src="https://github.com/survey-genomics/survey/blob/main/assets/logo_shine%405x.png" width="100px">
    </picture>
</p>
<p align="center">
</p>


The **Survey** toolkit is designed to process and analyze spatially-hashed single-cell data produced by the Survey Genomics assay. The core object is a MuData object ([docs](https://github.com/scverse/mudata), [repo](https://mudata.readthedocs.io/en/latest/#)) which stores single-cell data, spatial hash data, and any other modalities assayed concurrently (e.g. surface protein expression).


# System Requirements
Survey has been tested using Python 3.10 on MacOS and Ubuntu Linux.

# Installation

1. [Install mamba (or the latest conda).](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

2. Clone this repository and create a conda environment from the `environment.yml` file:

    ```
    mamba env create -f environment.yml
    mamba activate survey-env
    ```

3. Then, install the package using pip:

    ```
    pip install .
    ```



# Support

Please report issues or feature requests via GitHub.
