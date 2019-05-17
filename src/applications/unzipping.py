import operator

from utils.graph_clustering import clusterization_solver
from .measures.stress_points import stress_points
from applications.trainers import Naive


def unzip(hydra,
          losses,
          metrics,
          loaders,
          device,
          from_epoch,
          times=1,
          epochs=1):
    """
    Unzipping a hydra: finds the most stressful point (using the
    `applications.measures.stress_points` method; then, solve the
    graph clusterization problem using spectral methods; finally,
    split the hydra and fine-tune it for few epochs.

    Args:
      hydra:       an instance of Hydra class
      losses:      losses to fine-tune the model, a dict {task_id: loss_fn}
      metrics:     metrics to evaluate the model, a dict {task_id: metric_fn}
      loaders:     data loaders, instance of `torch.utils.data.DataLoader`
      device:      device to do training on
      from_epoch:  last epoch to continue from
      times:       number of times you want to repeat "calc-unzip-tune"
      epochs:      number of epochs to fine tune after each unzipping
    """
    for one_more_time in range(times):
        inner_measures, outer_measures = stress_points(
                hydra, losses, metrics, loaders, device)
        stressful_branch_index = max(
                inner_measures.items(), key=operator.itemgetter(1))[0]

        controller = hydra.controllers[stressful_branch_index]
        serving_tasks = list(controller.serving_tasks.keys())
        task_groups = []
        for child_index in controller.children_indices:
            child_controller = hydra.controllers[child_index]
            task_groups.append(
                    list(child_controller.serving_tasks.keys()))

        task_clusters = clusterization_solver(
                outer_measures[stressful_branch_index],
                serving_tasks, task_groups)

        branching_scheme = [set() for _ in range(len(task_clusters))]
        for cluster_idx in range(len(task_clusters)):
            for task_id in task_clusters[cluster_idx]:
                for child_index in controller.children_indices:
                    child_controller = hydra.controllers[child_index]
                    if task_id in child_controller.serving_tasks:
                        branching_scheme[cluster_idx].add(child_index)

        for i in range(len(branching_scheme)):
            branching_scheme[i] = list(branching_scheme[i])
        hydra.split(stressful_branch_index, branching_scheme, device)


        fine_tuner = Naive(
                device=device, model=hydra, losses=losses,
                metrics=metrics, train_loaders=loaders)
        for epoch in range(epochs):
            fine_tuner.train_epoch()
        del fine_tuner
