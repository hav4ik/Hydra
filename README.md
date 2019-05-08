
# Hydra &mdash; a Multi-Task Learning Framework

[![Using PyTorch 1.1](https://img.shields.io/badge/PyTorch%201.1-red.svg?labelColor=f3f4f7&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjU2cHgiIGhlaWdodD0iMzEwcHgiIHZpZXdCb3g9IjAgMCAyNTYgMzEwIiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIj4KICAgIDxnPgogICAgICAgIDxwYXRoIGQ9Ik0yMTguMjgxMDM3LDkwLjEwNjQxMiBDMjY4LjU3Mjk4OCwxNDAuMzk4MzYzIDI2OC41NzI5ODgsMjIxLjA3NTAzNCAyMTguMjgxMDM3LDI3MS43MTYyMzUgQzE2OS4wMzY4MzUsMzIyLjAwODE4NiA4OC4wMTA5MTQxLDMyMi4wMDgxODYgMzcuNzE4OTYzMiwyNzEuNzE2MjM1IEMtMTIuNTcyOTg3NywyMjEuNDI0Mjg0IC0xMi41NzI5ODc3LDE0MC4zOTgzNjMgMzcuNzE4OTYzMiw5MC4xMDY0MTIgTDEyNy44MjUzNzUsMCBMMTI3LjgyNTM3NSw0NS4wNTMyMDYgTDExOS40NDMzODMsNTMuNDM1MTk3OCBMNTkuNzIxNjkxNywxMTMuMTU2ODg5IEMyMi4wMDI3Mjg1LDE1MC4xNzczNTMgMjIuMDAyNzI4NSwyMTAuOTQ2Nzk0IDU5LjcyMTY5MTcsMjQ4LjY2NTc1NyBDOTYuNzQyMTU1NSwyODYuMzg0NzIgMTU3LjUxMTU5NiwyODYuMzg0NzIgMTk1LjIzMDU1OSwyNDguNjY1NzU3IEMyMzIuOTQ5NTIzLDIxMS42NDUyOTMgMjMyLjk0OTUyMywxNTAuODc1ODUzIDE5NS4yMzA1NTksMTEzLjE1Njg4OSBMMjE4LjI4MTAzNyw5MC4xMDY0MTIgWiBNMTczLjIyNzgzMSw4NC41MTg0MTc1IEMxNjMuOTY5MzM4LDg0LjUxODQxNzUgMTU2LjQ2Mzg0Nyw3Ny4wMTI5MjYzIDE1Ni40NjM4NDcsNjcuNzU0NDMzOCBDMTU2LjQ2Mzg0Nyw1OC40OTU5NDEzIDE2My45NjkzMzgsNTAuOTkwNDUwMiAxNzMuMjI3ODMxLDUwLjk5MDQ1MDIgQzE4Mi40ODYzMjMsNTAuOTkwNDUwMiAxODkuOTkxODE0LDU4LjQ5NTk0MTMgMTg5Ljk5MTgxNCw2Ny43NTQ0MzM4IEMxODkuOTkxODE0LDc3LjAxMjkyNjMgMTgyLjQ4NjMyMyw4NC41MTg0MTc1IDE3My4yMjc4MzEsODQuNTE4NDE3NSBaIiBmaWxsPSIjRUU0QzJDIj48L3BhdGg+CiAgICA8L2c+Cjwvc3ZnPgo=)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hav4ik/Hydra/blob/master/LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


[Hydra][hydra] is a flexible multi-task learning framework written on [PyTorch 1.0][pytorch]. The following multi-objective optimization algorithms are implemented:

* **Naive** &mdash; a separate optimizer for each task
* **Gradients averaging** &mdash; average out the gradients to the network's body
* **MGDA** &mdash; described in the paper [Multi-Task Learning as Multi-Objective Optimization (NIPS 2018)][mgda]

A comprehensive survey on these algorithms (and more) can be found in [this blog article][blog-post].

# Installation

* The code was written on `Python 3.6`. Clone this repository:

      git clone https://github.com/hav4ik/Hydra

* It is recommended to use [anaconda][conda] for installation of core packages (since `conda` packages comes with low-level libraries that can optimize the runtime):

      conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
      conda install numpy pandas

* Some of the packages are not available from anaconda, so you can install them using `pip`:

      pip install -r requirements.txt

# Getting started

* Examples of configuration files can be found [here][configs-dir]. A minimal example is available in [starter.sh][starter]. Execute it as follows (will train with configurations in [configs/toy_experiments/naive.yaml][naive-yaml]):

      ./starter.sh naive 50

# Coming soon...

* Neural Architecture Search for Multi-Task Learning.
* Proper framework documentation and examples.
* Built-in distillation from a bunch of single-task networks.


[hydra]: https://github.com/hav4ik/Hydra
[conda]: https://docs.conda.io/en/latest/miniconda.html
[pytorch]: https://pytorch.org/
[mgda]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[gradnorm]: https://arxiv.org/abs/1711.02257
[starter]: https://github.com/hav4ik/Hydra/blob/master/starter.sh
[configs-dir]: https://github.com/hav4ik/Hydra/tree/master/configs
[naive-yaml]: https://github.com/hav4ik/Hydra/blob/master/configs/toy_experiments/naive.yaml
[blog-post]: https://hav4ik.github.io/articles/mtl-a-practical-survey
