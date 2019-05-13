import sys
from tqdm import tqdm
import torch

from utils import log_utils


@torch.no_grad()
def kullback_leibner(p, q, batch=False):
    """Calculates Kullback-Leibner Divergence of two probability densities

    Args:
      p, q:   density values (must be all in range [0..1])
      batch:  if True, will return summed KL, divided by batch shape

    Returns:
      KL divergence, summed and divided by batch size if `batch` is `True`
    """
    kl = p * torch.log(p / q)
    if batch:
        kl = torch.sum(kl, 0) / kl.shape[0]
    return kl


@torch.no_grad()
def jensen_shannon(p, q, batch=False):
    """Calculates Jensen-Shannon Divergence of two probability densities

    Args:
      p, q:   density values (must be all in range [0..1])
      batch:  if True, will return summed KL, divided by batch shape

    Returns:
      JSD divergence, summed and divided by batch size if `batch` is `True`
    """
    m = (p + q) / 2.
    jsd = (kullback_leibner(p, m) + kullback_leibner(q, m)) / 2.
    if batch:
        jsd = torch.sum(jsd, 0) / jsd.shape[0]
    return jsd


@torch.no_grad()
def feature_similarity(hydras,
                       measure_requests,
                       loaders,
                       device=None,
                       compression='sigmoid',
                       measure='jensen_shannon'):
    """
    Given a list of hydras, calculates the similarity measure of their
    corresponding feature maps.

    WARNING: under current implementation, it is CRUCIAL that the last layer
             of the underlying block is BATCHNORM. This ensures that we can
             safely apply the Sigmoid function after that (for which we will
             calculate the divergence)

    Args:
      hydras:            a list of Hydra instances
      measure_requests:  a list of tuples (i, j, repi_id, repj_id, task_ids)
      loaders:           an dict of {task_id: torch.utils.data.DataLoader}
      device:            device to run everything on
      compression:       compression function, either `None` or 'sigmoid'
      measure:           measure, either jensen_shannon or kullback_leibner

    Raises:
      ValueError:          if some of requests in measure_requests are invalid
      NotImplementedError: if specified invalid `compression` or `measure`

    Returns:
      a list of measures for each of the request in measure_requests
    """
    log_utils.print_on_measure_begin()
    measurements = [None for _ in range(len(measure_requests))]
    task_ids = loaders.keys()
    for hydra in hydras:
        hydra.eval()

    # set BatchNorm Pillows to retain representations (need to unset later)
    # TODO: make this shit prettier!!! I'm so tired of this!!!
    # Aaaaalso, this will ONLY work if you do `BaseTrainer.warmup` first :')
    for hydra in hydras:
        for block in hydra.blocks:
            if hasattr(block, 'bn_pillow'):
                block.bn_pillow.retain_rep = True

    total_batches = sum([len(loader) for _, loader in loaders.items()])
    pbar = tqdm(desc='  eval ', total=total_batches, ascii=True)

    for task_id in task_ids:
        loader = loaders[task_id]
        task_measurements = [None for _ in range(len(measure_requests))]

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            for hydra in hydras:
                hydra(data, list(hydra.heads.keys()), retain_all=True)

            for buf_id, request in enumerate(measure_requests):
                i, j = request[0], request[1]
                repid_i, repid_j = request[2], request[3]
                if request[4] is not None:
                    if task_id not in request[4]:
                        continue

                hydra_i, hydra_j = hydras[i], hydras[j]
                if repid_i not in hydra_i.rep_tensors.keys() \
                        or repid_j not in hydra_j.rep_tensors.keys():
                    raise ValueError('matching_ids should contain valid '
                                     'indices of hydra''s rep_tensors.')

                # OH YEAH LOOK AT THIS UGLINESS OMG OMG OMG
                # TODO: make this prettier somehow

                if repid_i >= len(hydra_i.blocks) \
                        or repid_j >= len(hydra_j.blocks):
                    raise ValueError('matching_ids should contain valid '
                                     'indices of hydra''s blocks.')

                block_i = hydra_i.blocks[repid_i]
                block_j = hydra_j.blocks[repid_j]
                if not hasattr(block_i, 'bn_pillow') \
                        or not hasattr(block_j, 'bn_pillow'):
                    raise ValueError('matching_ids should only contain '
                                     'blocks with bn_pillow.')

                rep_i = block_i.bn_pillow.rep
                rep_j = block_j.bn_pillow.rep

                # PHEW, THIS UGLY STUFF IS OVER...
                # TODO: but seriously, we need to fix it!

                if not rep_i.shape == rep_j.shape:
                    raise ValueError(
                            "Representation tensor's shapes should be "
                            "equal, but got {} and {}.".format(
                                    str(rep_i.shape), str(rep_j.shape)))

                r = torch.randint(0, rep_i.shape[2], ())
                c = torch.randint(0, rep_i.shape[3], ())
                rep_i, rep_j = rep_i[:, :, r, c], rep_j[:, :, r, c]

                if compression == 'sigmoid':
                    rep_i, rep_j = torch.sigmoid(rep_i), torch.sigmoid(rep_j)
                else:
                    raise NotImplementedError(
                            '%s compression is not implemented' % compression)

                if not hasattr(sys.modules[__name__], measure):
                    raise NotImplementedError(
                            '%s measure is not implemented' % measure)
                measure_func = getattr(sys.modules[__name__], measure)
                d = measure_func(rep_i, rep_j, batch=True)

                if task_measurements[buf_id] is None:
                    task_measurements[buf_id] = d
                else:
                    task_measurements[buf_id] += d
            pbar.update()

        for buf_id, _ in enumerate(measure_requests):
            if task_measurements[buf_id] is not None:
                task_measurements[buf_id] /= \
                    torch.tensor(len(loader), device=device)
                if measurements[buf_id] is None:
                    measurements[buf_id] = task_measurements[buf_id]
                else:
                    measurements[buf_id] += task_measurements[buf_id]

    # unset BatchNorm Pillows (don't want to retain represnetations anymore)
    # TODO: make this shit prettier!!! I'm so tired of this!!!
    # Aaaaalso, this will ONLY work if you do `BaseTrainer.warmup` first :')
    for hydra in hydras:
        for block in hydra.blocks:
            if hasattr(block, 'bn_pillow'):
                block.bn_pillow.retain_rep = False

    return measurements
