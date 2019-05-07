from tqdm import tqdm
import torch
import torch.optim as optim

from .base_trainer import BaseTrainer
from utils.grad_normalizers import normalize_grads
from utils.min_norm_solver import MinNormSolver


class MGDA(BaseTrainer):
    def __init__(self,
                 device,
                 model,
                 model_manager,
                 task_ids,
                 losses,
                 metrics,
                 train_loaders,
                 test_loaders,
                 tensorboard_writer,
                 optimizers,
                 mode='phase_2',
                 normalize='loss+',
                 patience=None):

        super().__init__(
                device, model, model_manager, task_ids, losses, metrics,
                train_loaders, test_loaders, tensorboard_writer, patience)

        # Load Optimizers
        optimizer_def = getattr(optim, optimizers['method'])
        self.optimizer = optimizer_def(
                self.model.parameters(), **optimizers['kwargs'])
        self.grad_solver = [None] * len(model.blocks)
        self.temp_grad = [None] * len(model.blocks)
        for idx, (controller, block) in enumerate(model.control_blocks()):
            self.grad_solver[idx] = MinNormSolver(
                    len(controller.serving_tasks)).to(device)
        self.mode = mode
        self.normalize = normalize

    def train_epoch(self):
        """Trains the model on all data loaders for an epoch.
        """
        self.model.train()
        loader_iterators = dict([(k, iter(v))
                                 for k, v in self.train_loaders.items()])
        train_losses_ts = dict(
                [(k, torch.tensor(0.).to(self.device))
                 for k in self.task_ids])
        train_metrics_ts = dict(
                [(k, torch.tensor(0.).to(self.device))
                 for k in self.task_ids])
        total_batches = min([len(loader)
                             for _, loader in self.train_loaders.items()])
        pareto_count = 0
        if self.normalize is not None:
            loss_log = torch.empty(len(self.task_ids), device=self.device)

        pbar = tqdm(desc='  train', total=total_batches, ascii=True)
        for batch_idx in range(total_batches):
            # for each task, calculate head grads and accumulate body grads
            for task_idx, task_id in enumerate(self.task_ids):
                data, target = loader_iterators[task_id].next()
                data, target = data.to(self.device), target.to(self.device)

                # prepare grads
                self.model.zero_grad()

                # do inference with backward
                output = self.model(data, task_id)
                loss = self.losses[task_id](output, target)
                loss.backward()
                if self.normalize is not None:
                    loss_log[task_idx] = loss

                # save the body grads to temp_body_grad
                with torch.no_grad():
                    for ctrl, block in self.model.control_blocks(task_id):
                        if self.temp_grad[ctrl.index] is None:
                            self.temp_grad[ctrl.index] = []
                            for p in block.parameters():
                                self.temp_grad[ctrl.index].append(torch.empty(
                                    len(ctrl.serving_tasks), p.grad.numel(),
                                    device=self.device))

                        for i, p in enumerate(block.parameters()):
                            grad_idx = ctrl.serving_tasks[task_id]
                            self.temp_grad[ctrl.index][i][grad_idx] = \
                                p.grad.view(p.grad.numel())

                # calculate training metrics
                with torch.no_grad():
                    train_losses_ts[task_id] += loss.sum()
                    train_metrics_ts[task_id] += \
                        self.metrics[task_id](output, target)

            # Normalize grads and solving the min-norm problem
            with torch.no_grad():
                for idx, (c, block) in enumerate(self.model.control_blocks()):
                    assert idx == c.index
                    loss_idx = [i for i, k in enumerate(self.task_ids)
                                if k in c.serving_tasks]
                    for i, p in enumerate(block.parameters()):
                        if self.temp_grad[idx][i].shape[0] > 1:
                            if self.normalize is not None:
                                self.temp_grad[idx][i] = normalize_grads(
                                        self.temp_grad[idx][i],
                                        loss_log[loss_idx],
                                        self.normalize)
                            if self.mode == 'phase_2':
                                sol = self.grad_solver[idx](
                                        self.temp_grad[idx][i])
                                grad_star = torch.matmul(
                                        sol.unsqueeze_(0),
                                        self.temp_grad[idx][i])
                            else:
                                grad_star = self.temp_grad[idx][i].mean(0)
                        else:
                            grad_star = self.temp_grad[idx][i]

                        if torch.max(torch.abs(grad_star)) < 1e-5:
                            pareto_count += 1
                        p.grad.copy_(grad_star.view(p.grad.shape))

            self.optimizer.step()
            pbar.update()

        with torch.no_grad():
            for task_id in self.task_ids:
                train_losses_ts[task_id] /= \
                    len(self.train_loaders[task_id].dataset)
                train_metrics_ts[task_id] /= \
                    len(self.train_loaders[task_id].dataset)

        train_losses = dict([(k, v.item())
                             for k, v in train_losses_ts.items()])
        train_metrics = dict([(k, v.item())
                              for k, v in train_metrics_ts.items()])
        pbar.close()
        return train_losses, train_metrics
