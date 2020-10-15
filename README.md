```
              _                 _      
  /\/\   ___ | | ___  ___ _   _| | ___ 
 /    \ / _ \| |/ _ \/ __| | | | |/ _ \
/ /\/\ \ (_) | |  __/ (__| |_| | |  __/
\/    \/\___/|_|\___|\___|\__,_|_|\___|
   __           _                                      _   
  /__\ ____   _(_)_ __ ___  _ __  _ __ ___   ___ _ __ | |_ 
 /_\| '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|
//__| | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_ 
\__/|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__|

```                                                        
![testing](https://github.com/robmacc/capstone-molecule-environment/workflows/testing/badge.svg)

Reinforcement learning environment for inverse drug design.

## Set-up

Install dependencies:
```
conda env create -f environment.yml
```
Activate environment:
```
conda activate mol-env
```

## Getting started
To get a working gym environment all that's needed is to use the provided repository structure 
(see [here](https://github.com/openai/gym/blob/master/docs/creating-environments.md)):

* Dependencies that the environment needs have been defined in `setup.py`. 
* The environment's entry point has been defined in `gym_molecule/__init__.py`
* The environment has been imported into `gym_molecule/envs/__init.py__`
* With this structure the environment can be installed with `pip install -e .`
from the working directory.
* The environment definition is written in `gym_molecule/envs/molecule_env`,
and implements the interface provided by the `gym.Env` class (see 
the definition [here](https://github.com/openai/gym/blob/master/gym/core.py)).
* The essential methods defined are `step, reset, render, seed,`
and `close`.

## Installation
* Install anaconda python and pip. 
* Clone the repository from Gitlab/Github.
* Navigate into the cloned repository (directory)
* Run
 ```conda env create -f environment.yml```
* Run ```conda activate mol-env```
* Run ```pip install -e .```
* Run ```conda install -c schrodinger pymol-bundle```

The above was tested on a Windows 10 machine. The render method seemed to break on a VM running Ubuntu, and we didn't get to test the above instructions on a fully fledged Linux machine.

## Documentation Generation
This assumes you have the anaconda environment set-up, and will generate documentation using Sphinx.

* Make sure the anaconda environment mol-env is active. In Windows, run Anaconda prompt and run the command: ```conda activate mol-env``` in the project directory.
* Create a docs directory in the project base directory.
* Run ```sphinx-quickstart``` in the terminal/anaconda prompt. When prompted whether to separate source and build path, select `y`. 
The next 2 prompts of project and author name will not influence the document generation process and can take arbitrary values.
This will do some basic setup of the Sphinx documentation program and create some files, which include `conf.py` and `index.rst`. 
You will need to modify these files.
* In the file `conf.py`, uncomment and change the lines 
```
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
``` 

to

``` 
import os\\
import sys\\
sys.path.insert(0, os.path.abspath('../..'))
```
* Modify the ```extensions``` array in the `conf.py` file to ```extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon'].```

* Add the line 
```
.. automodule:: gym_molecule.envs.molecule_env 
    :members: 
```
to the index.rst file, above the ```.. toctree:: ``` line
* Run ```make html``` to generate the HTML version of the documentation. The documentation will be found in `docs/build/html`, and can be viewed in `index.html`.

## Testing
Pytest was used to test the environment, the user can run the tests from the project root directory.
