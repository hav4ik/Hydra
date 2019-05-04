from tqdm import tqdm
import torch
import torch.optim as optim

from .base_trainer import BaseTrainer


class GradNorm(BaseTrainer):
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
                 alpha=1.):

        super().__init__(
                device, model, model_manager, task_ids, losses, metrics,
                train_loaders, test_loaders, tensorboard_writer)

        self.coeffs = torch.ones(
                len(task_ids), requires_grad=True, device=device)
        optimizer_def = getattr(optim, optimizers['method'])
        self.model_optimizer = optimizer_def(
                model.parameters(), **optimizers['kwargs'])
        self.grad_optimizer = optimizer_def(
                [self.coeffs], **optimizers['kwargs'])

        self.has_loss_zero = False
        self.loss_zero = torch.empty(len(task_ids), device=device)
        self.alpha = torch.tensor(alpha, device=device)

    def train_epoch(self):
        """Trains the model on all data loaders for an epoch.
        """
        self.model.train()
        loader_iterators = dict([(k, iter(v))
                                 for k, v in self.train_loaders.items()])
        train_losses_ts = dict(
                [(k, torch.tensor(0.).to(self.device)) for k in self.task_ids])
        train_metrics_ts = dict(
                [(k, torch.tensor(0.).to(self.device)) for k in self.task_ids])
        total_batches = min([len(loader)
                             for _, loader in self.train_loaders.items()])
        num_tasks = torch.tensor(len(self.task_ids)).to(self.device)

        pbar = tqdm(desc='  train', total=total_batches, ascii=True)
        for batch_idx in range(total_batches):
            self.model.zero_grad()

            # for each task, calculate head grads, do backprop, and accumulate
            # gradients norms
            relative_inverse = torch.empty(
                    len(self.task_ids), device=self.device)
            grad_norm = torch.empty(len(self.task_ids), device=self.device)

            for task_idx, task_id in enumerate(self.task_ids):
                data, target = loader_iterators[task_id].next()
                data, target = data.to(self.device), target.to(self.device)

                # do inference and accumulate losses
                rep = self.model.body(data)
                rep.retain_grad()
                weighted_rep = rep * self.coeffs[task_idx]
                output = self.model.heads[task_id](weighted_rep)
                loss = self.losses[task_id](output, target)
                loss.backward()

                # GradNorm relative inverse training rate accumulation
                with torch.no_grad():
                    if not self.has_loss_zero:
                        self.loss_zero[task_idx] = loss
                    relative_inverse[task_idx] = loss
                    grad_norm[task_idx] = torch.sqrt(
                            torch.sum(torch.pow(rep.grad, 2)))

                # calculate training metrics
                with torch.no_grad():
                    train_losses_ts[task_id] += loss.sum()
                    train_metrics_ts[task_id] += \
                        self.metrics[task_id](output, target)

            # optimize the model
            self.model_optimizer.step()

            # GradNorm calculate relative inverse and avg gradients norm
            self.has_loss_zero = True
            with torch.no_grad():
                relative_inverse /= self.loss_zero
                relative_inverse /= torch.mean(relative_inverse)
                mean_norm = torch.mean(grad_norm)
                targets = torch.pow(relative_inverse, self.alpha) * mean_norm

            # GradNorm optimize coefficients
            self.grad_optimizer.zero_grad()
            coeff_loss = torch.sum(torch.abs(
                self.coeffs * grad_norm - targets))
            coeff_loss.backward()
            self.grad_optimizer.step()

            with torch.no_grad():
                self.coeffs /= self.coeffs.sum()
                self.coeffs *= num_tasks
            pbar.update()

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
