# Getting started

In the following we introduce the steps required to get asi-core sky-imaging tools running.


## Setup of the software

1. Clone the asi_core/ sky_imaging repository
2. Optional: Create an environment in which the python software should be installed, e.g. use miniforge and the `environment.yml`
located in the repository root:

`conda env create -f environment.yml`

3. Install the asi_core package (in the environment):

`cd <path_to_sky_imaging_repository>`

`pip install -e .`

4. Go to the root directory of the asi_core/ sky_imaging repository and execute the tests to make sure the installation 
worked fine:

`pytest tests`
