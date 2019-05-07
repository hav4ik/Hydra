import os
import yaml
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque


class Controller:
    """
    Hydra's block controller. Stores information about its index in the
    blocks list, the execution chain (blocks that should be executed in
    order before this block), and the children blocks of this block.

    Attributes:
      index:             the index of this block in the Hydra.blocks
      execution_chain:   indices of blocks to be executed prior to this
      parent_index:      index (in Hydra.blocks) of the parent block
      children_indices:  indices (in Hydra.blocks) of the childrens
      task_id:           if this block is a head, stores the task_id
    """
    def __init__(self, index=None):
        self.index = index
        self.execution_chain = [index]
        self.parent_index = None
        self.children_indices = []
        self.task_id = None
        self.serving_tasks = dict()

    def stack_on(self, controller):
        """Stacks current controller on top of another controller"""
        prev_chain = controller.execution_chain.copy()
        self.execution_chain = prev_chain + [self.index]
        self.parent_index = controller.index
        controller.children_indices.append(self.index)
        return self

    def __str__(self):
        return '({}): parent={}, children={}, serving=[{}]'.format(
                self.index, self.parent_index, self.children_indices,
                ', '.join(str(task_id) for task_id in self.serving_tasks))

    def __repr__(self):
        return str(self)

    def serialize(self):
        """Serialize to ordinary python's dict object"""
        return self.__dict__

    def deserialize(self, serialized_controller):
        """Deserialize from a python's dict object"""
        for k, v in serialized_controller.items():
            setattr(self, k, v)
        return self


class Hydra(nn.Module):
    """
    A base class for all Multi-Task Neural Networks with hard-shared
    parameters and arbitrary branching schema.

    Attributes:
      blocks:            a `nn.ModuleList` of building blocks of Hydra
      controllers:       a list of controllers accompanying each block
      heads:             dictionary {task_id: index} of Hydra's heads
      rep_tensors:       stores the tensors at branching points
      branching_points:  indices of blocks with more than one children
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.controllers = list()
        self.heads = dict()
        self.rep_tensors = dict()
        self.branching_points = set()

    def add_block(self, module):
        """
        Registers a new Hydra block, automatically adds it to the
        self.blocks and the execution graph.

        Args:
          module: a `nn.Module` object

        Returns:
          a Controller object for newly added block
        """
        new_index = len(self.blocks)
        new_controller = Controller(new_index)
        self.blocks.append(module)
        self.controllers.append(new_controller)
        return new_controller

    def add_head(self, module, task_id):
        """
        Registers a new Hydra block as a "Head". Same as the method
        `register_block()`, but adds the controller to self.heads.

        Args:
          module:    a `nn.Module` object
          task_id:  an identifier of the task that the head is solving

        Returns:
          a Controller object for newly added block
        """
        new_controller = self.add_block(module)
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
        """
        Dynamicaly constructs an execution plan, given the identifiers
        of tasks that we want to perform.

        Args:
          task_ids:  an identifier, or list of identifiers of tasks

        Returns:
          execution_order: a list of indices of modules to be executed
          branching_ids:   indices of branching points
        """
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
        """
        Returns an iterator over module parameters. If task_ids
        are specified, returns an iterator only over the parameters
        that affects the outputs on those tasks.

        Args:
          recurse:  whether to yield the parameters of submodules
          task_ids: whether to yield only task-related parameters

        Yields:
          Parameter: module parameter
        """
        if task_ids is None:
            for param in super().parameters(recurse):
                yield param
        else:
            execution_order, _ = self.execution_plan(task_ids)
            for index in execution_order:
                for param in self.blocks[index].parameters():
                    yield param

    def control_blocks(self, task_ids=None):
        """
        Yields an iterator over the blocks. If `task_ids` are specified,
        only blocks flowing towards corresponding heads will be yielded.
        """
        if task_ids is None:
            for controller, block in zip(self.controllers, self.blocks):
                yield controller, block
        else:
            execution_order, _ = self.execution_plan(task_ids)
            for index in execution_order:
                yield self.controllers[index], self.blocks[index]

    def create_branch(self, index, branches, device=None):
        """
        Dynamically clones `self.blocks[index]`, and stacks the branches
        specified by `branches` on top of the newly cloned branch.

        [Before]                         [After]
                    __ ...........           -------O--- ...........
            index  /                        / index
        --O-------O--- branches[0]       --O          __ branches[0]
                   \__                      \ clone  /
                       branches[1]           -------O--- branches[1]

        Args:
          index:      index of the block to clone
          branches:   indices of block's children to stach on the clone
          device:     device to spawn the clone on, can be decided later

        Raises:
          ValueError: in case invalid `index` or `branches` are specified

        Returns:
          controller: controller object of the newly created branch
          block:      module of the newly created branch
        """
        if index in self.heads:
            raise ValueError("Cannot split Hydra's head.")
        controller = self.controllers[index]
        for b in branches:
            if b not in controller.children_indices:
                raise ValueError("Indices of branches should be in "
                                 "controller's chilred_indices.")
        are_equal = True
        for b in controller.children_indices:
            if b not in branches:
                are_equal = False
        if are_equal:
            return self.controllers[index], self.blocks[index]

        block = self.blocks[index]
        cloned_block = deepcopy(block)
        if device is not None:
            cloned_block = cloned_block.to(device)
        cloned_controller = deepcopy(controller)
        new_index = len(self.controllers)
        cloned_controller.index = new_index
        self.blocks.append(cloned_block)
        self.controllers.append(cloned_controller)

        if cloned_controller.parent_index is not None:
            parent = self.controllers[cloned_controller.parent_index]
            parent.children_indices.append(new_index)
        cloned_controller.execution_chain = [
            i if i != index else new_index
            for i in cloned_controller.execution_chain]

        controller_deque = deque()
        controller_deque.extend(branches)
        while len(controller_deque) > 0:
            tmp_index = controller_deque.popleft()
            tmp_controller = self.controllers[tmp_index]
            if tmp_controller.parent_index == index:
                tmp_controller.parent_index = new_index
            tmp_controller.execution_chain = [
                i if i != index else new_index
                for i in tmp_controller.execution_chain]
            controller_deque.extend(tmp_controller.children_indices)

        controller.children_indices = [
            i for i in controller.children_indices
            if i not in branches]
        cloned_controller.children_indices = branches

        controller.serving_tasks = dict()
        for i in controller.children_indices:
            tmp_controller = self.controllers[i]
            controller.serving_tasks.update(
                tmp_controller.serving_tasks)
        cloned_controller.serving_tasks = dict()
        for i in cloned_controller.children_indices:
            tmp_controller = self.controllers[i]
            cloned_controller.serving_tasks.update(
                tmp_controller.serving_tasks)

        self.rep_tensors.clear()
        _, self.branching_points = \
            self.execution_plan(list(self.heads.keys()))

        return cloned_controller, cloned_block

    def split(self, index, branching_scheme):
        """
        Splits a Hydra's block into several blocks, according to the
        `branching_scheme`. Results of `split(0, [[1], [2,3], [4,5]])`:

        | B |  (1) (2) (3) (4) (5)     | A |  (1) (2) (3) (4) (5)
        | E |   |   |   |   |   |      | F |   |   |   |   |   |
        | F |   +---+---|---+---+      | T |   |   |---+   |---+
        | O |          (0)             | E |  (0) (6)     (7)
        | R |           |              | R |   |   |       |
        | E |          (*)             |   |  (*)--+-------+

        Args:
          index:            index of the block to split
          branching_scheme: list of list of indices (as example above)

        Raises:
          ValueError:       in case invalid parameters are specified

        Returns:
          controllers:      list of controllers of splitted branches
          blocks:           list of blocks - the splitted branches
        """
        if index not in self.branching_points:
            raise ValueError("You can only split layers which indices "
                             "are in `Hydra.branching_points`.")

        controller = self.controllers[index]
        block = self.blocks[index]

        total_branches = set()
        for branch in branching_scheme:
            total_branches.update(set(branch))
        if not total_branches == set(controller.children_indices):
            raise ValueError("The union of the branches should be "
                             "equal to `controller.children_indices`.")

        for i in range(len(branching_scheme)):
            scheme_a = set(branching_scheme[i])
            for j in range(i + 1, len(branching_scheme)):
                scheme_b = set(branching_scheme[j])
                if not scheme_a.isdisjoint(scheme_b):
                    raise ValueError("The branching schemes should "
                                     "be disjoint to each other.")

        new_controllers, new_blocks = [controller], [block]
        for branch in branching_scheme[1:]:
            tmp_ctrl, tmp_block = self.create_branch(index, branch)
            new_controllers.append(tmp_ctrl)
            new_blocks.append(tmp_block)
        return new_controllers, new_blocks

    def build(self):
        """
        Builds the model. Calculates additional stuffs to make the Hydra
        truly powerful.
        """
        for _, head_index in self.heads.items():
            controller = self.controllers[head_index]
            task_id = controller.task_id
            for index in controller.execution_chain:
                idx = len(self.controllers[index].serving_tasks)
                self.controllers[index].serving_tasks[task_id] = idx
        _, self.branching_points = \
            self.execution_plan(list(self.heads.keys()))

    def forward(self,
                input_tensor, task_ids, retain_tensors=False):
        """
        Defines the computation performed at every call. Dynamically
        and automatically decides what to run and in what order.

        Args:
          input_tensor:  a common input for specified tasks
          task_ids:      identifiers of tasks to be executed

        Returns:
          A dictionary {task_id: output} of task-specific outputs
        """
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
            if retain_tensors and index in self.branching_points:
                self.rep_tensors[index] = x
            if controller.task_id is not None:
                outputs[controller.task_id] = x

        if isinstance(task_ids, str):
            return outputs[task_ids]
        return outputs

    def serialize(self):
        """Serializes the Hydra into dictionary objects.

        Returns:
          hydra_serial:  a dictionary of Hydra's parameters
          state_dict:    a state dict of `nn.Module` object
        """
        controller_serializations = [
            c.serialize() for c in self.controllers]
        hydra_serialization = {
            'controllers': controller_serializations,
            'heads': self.heads
        }
        return hydra_serialization, self.state_dict()

    def deserialize(self, hydra_serialization, state_dict):
        """Reads the Hydra from its serialized representation.

        Args:
          hydra_serial:  a dictionary of Hydra's parameters
          state_dict:    a state dict of `nn.Module` object

        Returns: self
        """
        self.controllers = [
            Controller().deserialize(c)
            for c in hydra_serialization['controllers']
        ]
        self.heads = hydra_serialization['heads']
        self.load_state_dict(state_dict)
        return self

    def save(self, basepath):
        """
        Saves the Hydra to disc. The hydra will be saved in two parts:
          * basepath.yaml  -- stores the Hydra's controllers and heads
          * basepath.pth   -- stores the Hydra's weights

        Args:
          basepath: a full path to file (without extension) to save to
        """
        serialized_hydra, state_dict = self.serialize()
        basepath = os.path.expanduser(basepath)
        yaml_path = basepath + '.yaml'
        with open(yaml_path, 'w') as outfile:
            yaml.dump(serialized_hydra, outfile)
        pth_path = basepath + '.pth'
        torch.save(state_dict, pth_path)

    def load(self, basepath):
        """
        Loads the Hydra from dist. This will try to find two files:
          * basepath.yaml  -- for the Hydra's controllers and heads
          * basepath.pth   -- for the Hydra's weights

        Returns: self
        """
        basepath = os.path.expanduser(basepath)
        yaml_path = basepath + '.yaml'
        with open(yaml_path, 'r') as stream:
            serialized_hydra = yaml.safe_load(stream)
        pth_path = basepath + '.pth'
        state_dict = torch.load(pth_path)
        return self.deserialize(serialized_hydra, state_dict)
