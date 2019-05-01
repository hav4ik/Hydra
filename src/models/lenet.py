import torch.nn as nn
import torch.nn.functional as F


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return x


class Head(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc2 = nn.Linear(500, n_classes)

    def forward(self, x):
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Lenet(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.body = Body()

        head_dict = dict()
        for head in heads:
            task_id = head['task_id']
            head_kwargs = head['kwargs']
            head_dict[task_id] = Head(**head_kwargs)
        self.heads = nn.ModuleDict(head_dict)

    def forward(self, x, task_id):
        x = self.body(x)
        x = self.heads[task_id](x)
        return x
