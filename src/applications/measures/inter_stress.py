import torch

from .feature_similarity import feature_similarity


@torch.no_grad()
def inter_stress(hydras,
                 measure_requests,
                 loaders,
                 device=None):
    """
    Given a list of hydras, calculates the "stress" between corresponding
    models at given feature maps.

    WARNING: It is absolutely CRUCIAL to stack BATCHNORM layers on top of
             each hydra's blocks, using the flag `with_bn_pillow=True` in
             the `Block` constructor.

    Args:
      hydras:            a list of Hydra instances
      measure_requests:  a list of tuples (i, j, repi_id, repj_id, task_ids)
      loaders:           an dict of {task_id: torch.utils.data.DataLoader}
      device:            device to run everything on

    Raises:
      ValueError:        if some of requests in measure_requests are invalid
      RuntimeError:      if the bn_pillows are not initialized yet

    Returns:
      a list of measures for each of the request in measure_requests
    """
    similarity_measure = feature_similarity(
            hydras, measure_requests, loaders, device)
    inter_stress_measures = [None for _ in range(len(measure_requests))]

    for request_id, request in enumerate(measure_requests):
        hydra_i, hydra_j = hydras[request[0]], hydras[request[1]]
        rep_id_i, rep_id_j = request[2], request[3]

        # Feature importance for i-th hydra at feature map rep_id_i
        if not hydra_i.blocks[rep_id_i].with_bn_pillow:
            gamma_i = torch.ones(
                    similarity_measure[request_id].shape, device=device)
        else:
            if not hasattr(hydra_i.blocks[rep_id_i], 'bn_pillow'):
                raise RuntimeError('The bn_pillow is not yet initialized')
            gamma_i = torch.abs(hydra_i.blocks[rep_id_i].bn_pillow.weight)
        gamma_i = gamma_i / torch.sum(gamma_i)

        # Feature importance for j-th hydra at feature map rep_id_j
        if not hydra_j.blocks[rep_id_j].with_bn_pillow:
            gamma_j = torch.ones(
                    similarity_measure[request_id].shape, device=device)
        else:
            if not hasattr(hydra_j.blocks[rep_id_j], 'bn_pillow'):
                raise RuntimeError('The bn_pillow is not yet initialized')
            gamma_j = torch.abs(hydra_j.blocks[rep_id_j].bn_pillow.weight)
        gamma_j = gamma_j / torch.sum(gamma_j)

        # Now we have feature importance gamma_i and gamma_j that sums up
        # to 1; we also have per-channel feature similarities (JSD).
        inter_stress_measures[request_id] = \
            torch.sum((gamma_i + gamma_j) * similarity_measure[request_id])

    return inter_stress_measures
