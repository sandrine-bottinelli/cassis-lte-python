# cassis-lte-python

## About the project

The goal of this project is to provide python scripts 
to perform LTE modeling using the same philosophy
as in [CASSIS](http://cassis.irap.omp.eu).

## Table of content
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Contact us](#contact-us)

---


## Requirements

We recommend you to create a dedicated virtual
environment in which you can install the required packages.

### Using conda

#### Using conda and a .yml file
``` 
conda env create -f environment.yml
```
The name of the environment is set by the first line in the .yml file.
To choose your own name, you can either edit this file or use the following : 
``` 
conda env create -f environment.yml -n env_name
```
Note : the python version is specified in the .yml file.

To [update the environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment) : 
``` 
conda env update --file environment.yml  --prune
```

#### Using conda and the requirements.txt file
``` 
conda create --name env_name --file=requirements.txt
```
where env_name is the name of the environment.

Note : If you want to create an environment with a specific Python version, 
add ``python=<version>`` to the above command,
replacing ``<version>`` by the Python version, for example : 
``` 
conda create --name env_name --file=requirements.txt python=3.10
```

#### Remarks
- In both cases, activate your environment by typing `conda activate env_name`.
- 2023-12-07 : the latest versions of the packages spectral-cube and radio-beam
that are available on conda-forge (resp. 0.6.2 and 0.3.4) are not compatible with astropy=6.0,
which is why the provided requirements specify a lower version of astropy ;
if you need to update to astropy=6.0, you will have to install versions 0.6.3 and 0.3.5, 
respectively, of those two packages with pip (see next point).
- If you need to use pip, do it only after installing as many requirements as possible with conda. 
From the [conda user guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment) : 
once pip has been used, conda will be unaware of the changes ; 
to install additional conda packages, it is best to recreate the environment.
- For more information and details on the conda commands, see the [conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html).


### Using pip

* Create a new virtual environment : 
```
python3 -m venv env_name
```
where env_name is the name of the environment.
* Activate it :
```
source env_name/bin/activate
```
* Make sure pip is up-to-date and install requirements :
``` 
python3 -m pip install --upgrade pip
python3 -m pip install requirements.txt
```


## Getting started

Make sure that cassis-lte-python is in your PYTHONPATH.

To use your own settings :
- duplicate the config_defaults.ini
file in the cassis_lte_python directory,
- rename it 'config.ini' 
- edit it as necessary.

<!--- Example scripts for model creation are provided,
along with expected outputs : 
- ltm_example.py : generates a single model with one component and two species.
- ltm_example_grid1.py : computes models with one component for a grid of parameters.
- ltm_example_grid2.py : computes models with two components for a grid of parameters.
--->
## Contact us

[CASSIS team](mailto:cassis@irap.omp.eu) / [Sandrine Bottinelli](mailto:sbottinelli@irap.omp.eu)
