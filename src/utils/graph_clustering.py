import numpy as np
import pandas as pd
from sklearn.cluster import spectral_clustering


def _spectral_solver(outer_stress,
                     serving_tasks,
                     task_groups,
                     n_clusters=2):
    """Graph clusterization using spectral methods.

    Args:
      outer_stress:   a list of tuples of outer measurements info:
                      [(task_id_i, task_id_j, stress_value)]
                      returned by `applications.measures.stress_points`.
      serving_tasks:  list of tasks that the current branch is serving.
      task_groups:    these are list of list of tasks of children nodes.
      n_clusters:     number of clusters to divide to.

    Returns:
      a numpy array of cluster indices of each group, e.g. [0, 1, 0]
    """
    task_id_idx = dict((k, i) for i, k in enumerate(serving_tasks))
    data = np.zeros((len(serving_tasks), len(serving_tasks)))
    for task_id_i, task_id_j, stressval in outer_stress:
        data[task_id_idx[task_id_i], task_id_idx[task_id_j]] = stressval
    df_tasks = pd.DataFrame(
            data=data, index=serving_tasks, columns=serving_tasks)

    data = np.zeros((len(task_groups), len(task_groups)))
    for gid_i in range(len(task_groups)):
        for gid_j in range(len(task_groups)):
            t = df_tasks.loc[task_groups[gid_i], task_groups[gid_j]]
            ij_stress = t.max(axis=1).mean()
            t = df_tasks.loc[task_groups[gid_j], task_groups[gid_i]]
            ji_stress = t.max(axis=1).mean()
            data[gid_i, gid_j] = (ij_stress + ji_stress) / 2.
    df_groups = pd.DataFrame(data=data)

    affinity = df_groups.values
    affinity = np.exp(-affinity / affinity.max())
    clusters = spectral_clustering(affinity, n_clusters=n_clusters)
    return clusters


def _random_solver(outer_stress,
                   serving_tasks,
                   task_groups,
                   n_clusters=2):
    """Graph clusterization using random choice.

    Args:
      outer_stress:   a list of tuples of outer measurements info:
                      [(task_id_i, task_id_j, stress_value)]
                      returned by `applications.measures.stress_points`.
      serving_tasks:  list of tasks that the current branch is serving.
      task_groups:    these are list of list of tasks of children nodes.
      n_clusters:     number of clusters to divide to.

    Returns:
      a numpy array of cluster indices of each group, e.g. [0, 1, 0]
    """

    clusters = np.random.randint(n_clusters, size=len(task_groups))
    while np.unique(clusters).shape[0] < n_clusters:
        clusters = np.random.randint(n_clusters, size=len(task_groups))
    return clusters


def clusterization_solver(outer_stress,
                          serving_tasks,
                          task_groups,
                          n_clusters=2,
                          method='spectral'):
    """Graph clusterization solver, using spectral or random methods

    Args:
      outer_stress:   a list of tuples of outer measurements info:
                      [(task_id_i, task_id_j, stress_value)]
                      returned by `applications.measures.stress_points`.
      serving_tasks:  list of tasks that the current branch is serving.
      task_groups:    these are list of list of tasks of children nodes.
      n_clusters:     number of clusters to divide to.
      method:         either 'spectral' or 'random'; others not supported.

    Raises:
      NotImplementedError:  In case invalid `method` was specified.

    Returns:
      a list of length `n_clusters` of lists of task_id for each cluster;
      1-1 matching can be performed between tasks and child nodes.
    """

    if method == 'spectral':
        clusters = _spectral_solver(
                outer_stress, serving_tasks, task_groups, n_clusters)
    elif method == 'random':
        clusters = _random_solver(
                outer_stress, serving_tasks, task_groups, n_clusters)
    else:
        raise NotImplementedError("Only 'spectral' and 'random' are "
                                  "supported.")

    task_clusters = [[] for _ in range(n_clusters)]
    for i, c in enumerate(clusters):
        task_clusters[c].extend(task_groups[i])
    return task_clusters
