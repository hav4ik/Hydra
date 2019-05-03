import sys
import torch.nn.functional as F


def corrects(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum()
    return correct


def get_metrics(metrics):
    """Get metrics for dictionary {task_id: metric}
    """
    metric_dict = dict()
    for k, v in metrics.items():
        if hasattr(F, v):
            metric_dict[k] = getattr(F, v)
        else:
            metric_dict[k] = getattr(sys.modules[__name__], v)
    return metric_dict
