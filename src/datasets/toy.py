import os
import torchvision


def toy(dataset,
        root='~/data/torchvision/',
        transforms=None):
    """Load a train and test datasets from torchvision.dataset.
    """
    if not hasattr(torchvision.datasets, dataset):
        raise ValueError
    loader_def = getattr(torchvision.datasets, dataset)

    composed_transform = None
    if transforms is not None:
        transform_funcs = []
        for transform in transforms:
            if not hasattr(torchvision.transforms, transform['def']):
                raise ValueError
            transform_def = getattr(torchvision.transforms, transform)
            transform_funcs.append(transform_def(**transform['kwargs']))
        composed_transform = torchvision.transforms.Compose(transform_funcs)

    train_loader = loader_def(root=os.path.expanduser(root), train=True,
                              download=True, transform=composed_transform)
    test_loader = loader_def(root=os.path.expanduser(root), train=False,
                             download=True, transform=composed_transform)
    return train_loader, test_loader
