import sys
import torch.nn.functional as F


def get_losses(losses):
    """Get loss functions for dictionary {task_id: loss}
    """
    loss_dict = dict()
    for k, v in losses.items():
        if hasattr(F, v):
            loss_dict[k] = getattr(F, v)
        else:
            loss_dict[k] = getattr(sys.modules[__name__], v)
    return loss_dict
