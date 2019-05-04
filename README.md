# Hydra &mdash; Multi-Task Learning Framework for PyTorch

A flexible and convenient framework for Multi-Task Learning. The following algorithms are implemented:

* Naive
* Gradients averaging
* MGDA

# Installation

* The code was written on `Python 3.6`. Clone this repository:

      git clone https://github.com/hav4ik/emtl

* It is recommended to use [anaconda][conda] for installation of core packages (since `conda` packages comes with low-level libraries that can optimize the runtime):

      conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
      conda install numpy pandas

* Some of the packages are not available from anaconda, so you can install them using `pip`:

      pip install -r requirements.txt


[conda]: https://docs.conda.io/en/latest/miniconda.html
