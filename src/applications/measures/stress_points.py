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
      a dict {branch_index: tensor} of stress values of the Hydra's block
    """
    peeled_hydras = split_tuning(hydra, losses, metrics, loaders, device)
    hydras = list(h for _, (h, _) in peeled_hydras.items())
    index_maps = list(m for _, (_, m) in peeled_hydras.items())
    task_ids = list(t for t, (_, _) in peeled_hydras.items())

    measure_requests = []
    hydras.append(hydra)
    this_idx = len(hydras) - 1

    for branching_index in hydra.branching_points:
        for idx in range(len(peeled_hydras)):
            index_map = index_maps[idx]
            task_id = task_ids[idx]
            if branching_index not in index_map:
                continue
            i, j = this_idx, idx
            rep_id_i, rep_id_j = branching_index, index_map[branching_index]
            measure_requests.append((i, j, rep_id_i, rep_id_j, task_id))

    inter_stress_measures = inter_stress(
            hydras, measure_requests, loaders, device)

    with torch.no_grad():
        branching_stress = dict(
                (k, torch.tensor(0., device=device))
                for k in hydra.branching_points)
        for idx in range(len(inter_stress_measures)):
            rep_id_i = measure_requests[idx][2]
            branching_stress[rep_id_i] += inter_stress_measures[idx]

    return branching_stress
