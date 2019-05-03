import torch


def normalize_grads(grads, losses, normalization_type):
    """Grads should be a 2D tensor of flattened gradients
    """
    if normalization_type == 'l2':
        gns = (grads.pow(2).sum(grads, 1)).sqrt()
    elif normalization_type == 'loss':
        gns = losses
    elif normalization_type == 'loss+':
        gns = losses * (grads.pow(2).sum(1)).sqrt()
    else:
        gns = torch.ones(grads.shape[0], device=grads.device)

    # according to documentation, the following are in-place
    transposed = torch.transpose(grads, 1, 0)
    transposed /= gns
    grads = torch.transpose(transposed, 1, 0)
    return grads
