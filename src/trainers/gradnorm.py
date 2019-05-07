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

        # Important stuff
        relative_inverse = torch.empty(
                len(self.task_ids), device=self.device)
        _, all_branching_ids = self.model.execution_plan(self.task_ids)
        grad_norm = dict([
                (k, torch.zeros(len(self.task_ids), device=self.device))
                for k in all_branching_ids])

        pbar = tqdm(desc='  train', total=total_batches, ascii=True)
        for batch_idx in range(total_batches):
            # Stuffs should be manually zeroed
            tmp_coeffs = self.coeffs.clone().detach()
            self.model.zero_grad()
            self.grad_optimizer.zero_grad()
            for k, v in self.model.rep_tensors.items():
                if v.grad is not None:
                    v.grad.zero_()
                if v is not None:
                    v.detach()

            # for each task, calculate head grads, do backprop, and accumulate
            # gradients norms
            for task_idx, task_id in enumerate(self.task_ids):
                data, target = loader_iterators[task_id].next()
                data, target = data.to(self.device), target.to(self.device)

                # do inference and accumulate losses
                output = self.model(data, task_id, retain_tensors=True)
                for index in all_branching_ids:
                    self.model.rep_tensors[index].retain_grad()
                loss = self.losses[task_id](output, target)
                weighted_loss = tmp_coeffs[task_idx] * loss

                # Not retaining graph since we don't need the values anymore.
                # Create a graph of gradients to use them later.
                weighted_loss.backward(retain_graph=False, create_graph=True)
                output.detach()

                # GradNorm relative inverse training rate accumulation
                if not self.has_loss_zero:
                    self.loss_zero[task_idx] = loss.clone().detach()
                relative_inverse[task_idx] = loss.clone().detach()

                # GradNorm accumulate gradients
                # wtf_loss = torch.tensor(0, device=self.device)
                for index in all_branching_ids:
                    grad = self.model.rep_tensors[index].grad
                    grad_norm[index][task_idx] = torch.sqrt(
                            torch.sum(torch.pow(grad, 2)))
                    # wtf_loss = wtf_loss + grad_norm[index][task_idx]
                # if task_idx == 1:
                #     wtf_loss.backward(retain_graph=True, create_graph=True)
                #     print('WTF 1', self.coeffs.grad.shape)

                # calculate training metrics
                with torch.no_grad():
                    train_losses_ts[task_id] += loss.sum()
                    train_metrics_ts[task_id] += \
                        self.metrics[task_id](output, target)

            # GradNorm calculate relative inverse and avg gradients norm
            self.has_loss_zero = True
            relative_inverse = relative_inverse / self.loss_zero.clone().detach()
            relative_inverse = relative_inverse / torch.mean(relative_inverse).clone().detach()
            relative_inverse = torch.pow(relative_inverse, self.alpha.clone().detach())

            # wtf_loss = torch.tensor(0, device=self.device)
            # wtf_loss = wtf_loss + grad_norm[2].mean() + relative_inverse.mean() + self.alpha
            # wtf_loss.backward()
            # print(self.coeffs.grad)
            # raise

            coeff_loss = torch.tensor(0., device=self.device)
            for k, rep_grads in grad_norm.items():
                mean_norm = torch.mean(rep_grads)
                target = relative_inverse * mean_norm
                # print('relainv', relative_inverse)
                # print('target', target, mean_norm)
                coeff_loss = coeff_loss + mean_norm.mean()
                # coeff_loss = coeff_loss + \
                #     torch.sum(torch.abs(rep_grads - target))

            # GradNorm optimize coefficients
            coeff_loss.backward()

            # self.coeffs.grad = tmp_coeffs.grad.copy().detach()
            # self.grad_optimizer.step()

            # optimize the model
            self.model_optimizer.step()



            # with torch.no_grad():
            #     self.coeffs /= self.coeffs.sum()
            #     self.coeffs *= num_tasks
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
