import torch.nn as nn
import torch.nn.functional as F

from .hydra_base import Hydra, Block


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=1,
                stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(x)
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
        else:
            shortcut = x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet18(Hydra):
    """
    Pre-Activation ResNet
    https://arxiv.org/abs/1603.05027
    """
    def __init__(self,
                 heads,
                 num_planes=[32, 64, 64, 128],
                 num_blocks=[2, 2, 2, 2]):
        super().__init__()
        self.in_planes = 32

        layers = [nn.Conv2d(
            1, self.in_planes, kernel_size=3,
            stride=1, padding=1, bias=False)]
        bn_planes = [self.in_planes]

        layers.extend(self._make_layer(num_planes[0], num_blocks[0], 1))
        layers.extend(self._make_layer(num_planes[1], num_blocks[1], 2))
        layers.extend(self._make_layer(num_planes[2], num_blocks[2], 2))
        layers.extend(self._make_layer(num_planes[3], num_blocks[3], 2))

        bn_planes.extend([num_planes[0]] * num_blocks[0])
        bn_planes.extend([num_planes[1]] * num_blocks[1])
        bn_planes.extend([num_planes[2]] * num_blocks[2])
        bn_planes.extend([num_planes[3]] * num_blocks[3])

        controller = self.add_block(
            Block(layers[0], bn_pillow_planes=bn_planes[0]))
        for layer, nplanes in zip(layers[1:], bn_planes[1:]):
            new_controller = self.add_block(
                Block(layer, bn_pillow_planes=nplanes)).stack_on(controller)
            controller = new_controller

        def define_head(n_classes):
            return Block(nn.Sequential(*[
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self.in_planes, n_classes),
                    nn.LogSoftmax(dim=1)]))

        for head in heads:
            module = define_head(head['n_classes'])
            h = self.add_head(module, head['task_id'])
            h.stack_on(controller)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return layers
