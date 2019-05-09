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

    transform_funcs = []
    if transforms is not None:
        for transform in transforms:
            if not hasattr(torchvision.transforms, transform['def']):
                raise ValueError
            transform_def = getattr(torchvision.transforms, transform['def'])
            transform_funcs.append(transform_def(**transform['kwargs']))
    transform_funcs.append(torchvision.transforms.ToTensor())

    composed_transform = torchvision.transforms.Compose(transform_funcs)
    trainset = loader_def(
            root=os.path.expanduser(root), train=True,
            download=True, transform=composed_transform)
    testset = loader_def(
            root=os.path.expanduser(root), train=False,
            download=True, transform=composed_transform)
    return trainset, testset
