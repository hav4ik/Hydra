import os
import yaml
import torch
import torch.nn as nn


class Controller:
    def __init__(self, index=None):
        self.index = index
        self.execution_chain = [index]
        self.parent_index = None
        self.children_indices = []
        self.task_id = None

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

    def serialize(self):
        return self.__dict__

    def deserialize(self, serialized_controller):
        for k, v in serialized_controller.items():
            setattr(self, k, v)
        return self


class Hydra(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.controllers = list()
        self.heads = dict()
        self.rep_tensors = dict()

    def register_block(self, block):
        new_index = len(self.blocks)
        new_controller = Controller(new_index)
        self.blocks.append(block)
        self.controllers.append(new_controller)
        return new_controller

    def register_head(self, block, task_id):
        new_controller = self.register_block(block)
        new_controller.task_id = task_id
        self.heads[task_id] = new_controller.index
        return new_controller

    def extra_repr(self):
        items = '\n  '.join(str(c) for c in self.controllers)
        controllers = '(block controllers):\n  ' + items
        items = '\n  '.join(
                '({}) -> {}'.format(k, str(c))
                for k, c in self.heads.items())
        heads = '(heads):\n  ' + items
        return controllers + '\n' + heads

    def execution_plan(self, task_ids):
        if not isinstance(task_ids, list):
            task_ids = [task_ids]
        execution_order = []
        branching_ids = set()
        for task_id in task_ids:
            branching_point = None
            controller = self.controllers[self.heads[task_id]]
            task_exec_chain = controller.execution_chain
            for i, index in enumerate(task_exec_chain):
                if index not in execution_order:
                    break
                branching_point = index
            execution_order += task_exec_chain[i:].copy()
            if branching_point is not None:
                branching_ids.add(branching_point)
        return execution_order, branching_ids

    def parameters(self, recurse=True, task_ids=None):
        if task_ids is None:
            for param in super().parameters(recurse):
                yield param
        else:
            execution_order, _ = self.execution_plan(task_ids)
            for index in execution_order:
                for param in self.blocks[index].parameters():
                    yield param

    def forward(self, input_tensor, task_ids):
        exec_order, branching_ids = self.execution_plan(task_ids)
        x = input_tensor
        outputs = dict()
        for index in exec_order:
            controller = self.controllers[index]
            parent_index = controller.parent_index
            if parent_index not in branching_ids:
                x = self.blocks[index](x)
            else:
                x = self.blocks[index](self.rep_tensors[parent_index])
            if index in branching_ids:
                self.rep_tensors[index] = x
            if controller.task_id is not None:
                outputs[controller.task_id] = x
        return outputs

    def serialize(self):
        controller_serializations = [
            c.serialize() for c in self.controllers]
        hydra_serialization = {
            'controllers': controller_serializations,
            'heads': self.heads
        }
        return hydra_serialization, self.state_dict()

    def deserialize(self, hydra_serialization, state_dict):
        self.controllers = [
            Controller().deserialize(c)
            for c in hydra_serialization['controllers']
        ]
        self.heads = hydra_serialization['heads']
        self.load_state_dict(state_dict)
        return self

    def save(self, basepath):
        serialized_hydra, state_dict = self.serialize()
        basepath = os.path.expanduser(basepath)
        yaml_path = basepath + '.yaml'
        with open(yaml_path, 'w') as outfile:
            yaml.dump(serialized_hydra, outfile)
        pth_path = basepath + '.pth'
        torch.save(state_dict, pth_path)

    def load(self, basepath):
        basepath = os.path.expanduser(basepath)
        yaml_path = basepath + '.yaml'
        with open(yaml_path, 'r') as stream:
            serialized_hydra = yaml.safe_load(stream)
        pth_path = basepath + '.pth'
        state_dict = torch.load(pth_path)
        return self.deserialize(serialized_hydra, state_dict)
