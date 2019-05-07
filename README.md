# Hydra &mdash; a Multi-Task Learning Framework

[Hydra][hydra] is a flexible multi-task learning framework written on [PyTorch 1.0][pytorch]. The following multi-task learning algorithms are implemented:

* **Naive** &mdash; a separate optimizer for each task
* **Gradients averaging** &mdash; average out the gradients to the network's body
* **MGDA** &mdash; described in the paper [Multi-Task Learning as Multi-Objective Optimization (NIPS 2018)][mgda]

# Installation

* The code was written on `Python 3.6`. Clone this repository:

      git clone https://github.com/hav4ik/emtl

* It is recommended to use [anaconda][conda] for installation of core packages (since `conda` packages comes with low-level libraries that can optimize the runtime):

      conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
      conda install numpy pandas

* Some of the packages are not available from anaconda, so you can install them using `pip`:

      pip install -r requirements.txt


[hydra]: https://github.com/hav4ik/Hydra
[conda]: https://docs.conda.io/en/latest/miniconda.html
[pytorch]: https://pytorch.org/
[mgda]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[gradnorm]: https://arxiv.org/abs/1711.02257
