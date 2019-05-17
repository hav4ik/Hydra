import torch

from applications.trainers import Naive
from applications.measures.inter_stress import inter_stress
from models.hydra_base import Block


def split_tuning(hydra, losses, metrics, loaders, device, epochs=3):
    """
    Splits the Multi-Task Hydra into multiple single-task ones, and then
    fine-tune each of them for few epochs.

    Args:
      hydra:     the Multi-Task Hydra that we're gonna split
      losses:    losses to fine-tune the model, a dict {task_id: loss_fn}
      metrics:   metrics to evaluate the model, a dict {task_id: metric_fn}
      loaders:   data loaders, instance of `torch.utils.data.DataLoader`
      device:    device to do training on

    Returns:
      a dict {task_id: (single_task_hydra, index_map)} of pairs of fine-tuned
      sub-hydras and map of correspondence between original and peeled hydras
    """
    hydras = dict()
    for task_id in hydra.heads.keys():
        peeled_hydra, index_map = hydra.peel(task_id, device=device)
        for block_index, block in enumerate(peeled_hydra.blocks):
            if block_index not in peeled_hydra.branching_points:
                continue
            if isinstance(block, Block):
                block.with_bn_pillow = True

        tmp_trainer = Naive(
                device=device, model=peeled_hydra, losses=losses,
                metrics=metrics, train_loaders=loaders, slimming=0.0045)
        for epoch in range(epochs):
            tmp_trainer.train_epoch()
        hydras[task_id] = (peeled_hydra, index_map)

    return hydras


def stress_points(hydra, losses, metrics, loaders, device):
    """
    Finding stress points of a Hydra (on some specific data) among the
    branching layers (with BN Pillow).

    Args:
      hydra:     the Multi-Task Hydra that we're gonna split
      losses:    losses to fine-tune the model, a dict {task_id: loss_fn}
      metrics:   metrics to evaluate the model, a dict {task_id: metric_fn}
      loaders:   data loaders, instance of `torch.utils.data.DataLoader`
      device:    device to do training on

    Returns:
      inner_stress: a dict {branch_index: value} of branches stress values
      outer_stress: a dict of lists of tuples of outer measurements info:
                    {branch_index: [(task_id_i, task_id_j, stress_value)]}
    """
    peeled_hydras = split_tuning(hydra, losses, metrics, loaders, device)
    hydras = list(h for _, (h, _) in peeled_hydras.items())
    index_maps = list(m for _, (_, m) in peeled_hydras.items())
    task_ids = list(t for t, (_, _) in peeled_hydras.items())

    measure_requests = []
    hydras.append(hydra)
    this_idx = len(hydras) - 1

    # Measurements for stress inside each branching point
    for branching_index in hydra.branching_points:
        for idx in range(len(peeled_hydras)):
            index_map = index_maps[idx]
            task_id = task_ids[idx]
            if branching_index not in index_map:
                continue
            i, j = this_idx, idx
            rep_id_i, rep_id_j = branching_index, index_map[branching_index]
            measure_requests.append((i, j, rep_id_i, rep_id_j, task_id))

    # Now, the measurements for stress inside branches lies in the slice
    # measure_requests[:inter_request_len]
    innerstress_request_len = len(measure_requests)

    # Measurements of difference between each of the peeled hydras
    outer_stress = dict((k, []) for k in hydra.branching_points)
    for branching_index in hydra.branching_points:

        for idx_i in range(len(peeled_hydras)):
            index_map_i = index_maps[idx_i]
            task_id_i = task_ids[idx_i]
            if branching_index not in index_map_i:
                continue
            rep_id_i = index_map_i[branching_index]

            for idx_j in range(idx_i + 1, len(peeled_hydras)):
                index_map_j = index_maps[idx_j]
                task_id_j = task_ids[idx_j]
                if branching_index not in index_map_j:
                    continue
                rep_id_j = index_map_j[branching_index]

                # TODO: make this thing less ugly. We don't want to track
                # the tasks, we want to work ONLY with children indices.
                outer_stress[branching_index].append(
                        (task_id_i, task_id_j, len(measure_requests)))
                outer_stress[branching_index].append(
                        (task_id_j, task_id_i, len(measure_requests)))
                measure_requests.append(
                        (idx_i, idx_j, rep_id_i, rep_id_j, None))

    # The moment of truth...
    stress_measures = inter_stress(
            hydras, measure_requests, loaders, device)

    # Calculating inner stress of each branch
    with torch.no_grad():
        inner_stress = dict(
                (k, torch.tensor(0., device=device))
                for k in hydra.branching_points)
        for idx in range(innerstress_request_len):
            rep_id_i = measure_requests[idx][2]
            inner_stress[rep_id_i] += stress_measures[idx]
    for k in inner_stress.keys():
        inner_stress[k] = inner_stress[k].item()

    # Gathering the stresses between each subnetworks
    for k, v in outer_stress.items():
        for i in range(len(v)):
            task_id_i, task_id_j, idx = v[i]
            v[i] = (task_id_i, task_id_j, stress_measures[idx].item())

    del peeled_hydras
    return inner_stress, outer_stress
