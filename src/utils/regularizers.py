import torch
import torch.nn.functional as F


def slimming_loss(hydra):
    """
    Implementation of the work Liu. et. al. "Network Slimming" (ICCV'17)
    For more details refer to the paper https://arxiv.org/abs/1708.06519

    Only applied to Hydras that has blocks of Block instance with
    `with_bn_pillow == True` specified.
    """
    cum_loss = torch.tensor(0., device=next(hydra.parameters()).device)
    for block in hydra.blocks:
        if not block.with_bn_pillow:
            continue
        if not hasattr(block, 'bn_pillow'):
            raise RuntimeError('The bn_pillow is not yet initialized')

        w = block.bn_pillow.weight
        cum_loss = cum_loss + F.smooth_l1_loss(w, torch.zeros_like(w))
    return cum_loss
