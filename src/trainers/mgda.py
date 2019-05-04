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
                 normalize='loss+'):

        super().__init__(
                device, model, model_manager, task_ids, losses, metrics,
                train_loaders, test_loaders, tensorboard_writer)

        # Load Optimizers
        optimizer_def = getattr(optim, optimizers['method'])
        head_optimizers = dict()
        for task_id in task_ids:
            head_optimizers[task_id] = optimizer_def(
                    model.heads[task_id].parameters(),
                    **optimizers['kwargs'])
        body_optimizers = optimizer_def(
                model.body.parameters(), **optimizers['kwargs'])
        optimizers_dict = {
                'body': body_optimizers, 'head': head_optimizers}
        self.grad_solver = MinNormSolver(len(task_ids)).to(device)
        self.normalize = normalize
        self.temp_body_grad = None
        self.optimizers = optimizers_dict

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

        pbar = tqdm(desc='  train', total=total_batches, ascii=True)
        if self.normalize is not None:
            loss_log = torch.empty(len(self.task_ids), device=self.device)
        for batch_idx in range(total_batches):
            # for each task, calculate head grads and accumulate body grads
            for task_idx, task_id in enumerate(self.task_ids):
                data, target = loader_iterators[task_id].next()
                data, target = data.to(self.device), target.to(self.device)

                # prepare grads
                self.model.body.zero_grad()
                self.model.heads[task_id].zero_grad()

                # do inference with backward
                output = self.model(data, task_id=task_id)
                loss = self.losses[task_id](output, target)
                loss.backward()
                if self.normalize is not None:
                    loss_log[task_idx] = loss

                # optimize the heads right away
                self.optimizers['head'][task_id].step()

                # save the body grads to temp_body_grad
                with torch.no_grad():
                    if self.temp_body_grad is None:
                        self.temp_body_grad = []
                        for p in self.model.body.parameters():
                            self.temp_body_grad.append(torch.empty(
                                len(self.task_ids), p.grad.numel(),
                                device=self.device))
                    for i, p in enumerate(self.model.body.parameters()):
                        self.temp_body_grad[i][task_idx] = \
                            p.grad.view(p.grad.numel())

                # calculate training metrics
                with torch.no_grad():
                    train_losses_ts[task_id] += loss.sum()
                    train_metrics_ts[task_id] += \
                        self.metrics[task_id](output, target)

            # Averaging out body gradients and optimize the body
            with torch.no_grad():
                for i, p in enumerate(self.model.body.parameters()):
                    if self.normalize is not None:
                        self.temp_body_grad[i] = normalize_grads(
                                self.temp_body_grad[i], loss_log,
                                self.normalize)
                    sol = self.grad_solver(self.temp_body_grad[i])
                    grad_star = torch.matmul(
                            sol.unsqueeze_(0), self.temp_body_grad[i])
                    if torch.max(torch.abs(grad_star)) < 1e-5:
                        pareto_count += 1
                    p.grad.copy_(grad_star.view(p.grad.shape))

            self.optimizers['body'].step()
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
