# Hydra &mdash; a Multi-Task Learning and NAS Framework

[Hydra][hydra] is a flexible multi-task learning framework written on [PyTorch 1.0][pytorch]. The following multi-objective optimization algorithms are implemented:

* **Naive** &mdash; a separate optimizer for each task
* **Gradients averaging** &mdash; average out the gradients to the network's body
* **MGDA** &mdash; described in the paper [Multi-Task Learning as Multi-Objective Optimization (NIPS 2018)][mgda]

A comprehensive survey on these algorithms (and more) can be found at [hav4ik.github.io/articles/mtl-a-practical-survey][blog-post].

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
