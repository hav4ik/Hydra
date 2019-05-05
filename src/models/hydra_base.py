import torch.nn as nn


class Controller:
    def __init__(self, index):
        self.index = index
        self.execution_chain = [index]
        self.parent_index = None
        self.children_indices = []

    def stack_on(self, controller):
        prev_chain = controller.execution_chain.copy()
        self.execution_chain = prev_chain + [self.index]
        self.parent_index = controller.index
        controller.children_indices.append(self.index)

    def __str__(self):
        return '({}): parent={}, children={}'.format(
                self.index, self.parent_index,
                self.children_indices)

    def __repr__(self):
        return str(self)


class Hydra(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.controllers = list()
        self.heads = dict()

    def register(self, block):
        new_index = len(self.blocks)
        new_controller = Controller(new_index)
        self.blocks.append(block)
        self.controllers.append(new_controller)
        return new_controller

    def register_head(self, block, task_id):
        new_controller = self.register(block)
        self.heads[task_id] = new_controller
        return new_controller

    def extra_repr(self):
        items = '\n  '.join(str(c) for c in self.controllers)
        controllers = '(block controllers):\n  ' + items
        items = '\n  '.join(
                '{} -> {}'.format(k, str(c))
                for k, c in self.heads.items())
        heads = '(heads):\n  ' + items
        return controllers + '\n' + heads
