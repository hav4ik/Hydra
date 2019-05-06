from tqdm import tqdm
import torch

from utils import log_utils


class BaseTrainer:
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
                 **kwargs):

        self.device = device
        self.model = model
        self.model_manager = model_manager
        self.task_ids = task_ids
        self.losses = losses
        self.metrics = metrics
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.tensorboard_writer = tensorboard_writer

    def eval_epoch(self):
        self.model.eval()
        total_batches = sum([len(loader)
                            for _, loader in self.test_loaders.items()])
        eval_losses, eval_metrics = dict(), dict()

        pbar = tqdm(desc='  eval ', total=total_batches, ascii=True)
        with torch.no_grad():
            for task_id in self.task_ids:
                loader = self.test_loaders[task_id]
                loss = torch.tensor(0.).to(self.device)
                metric = torch.tensor(0.).to(self.device)
                for batch_idx, (data, target) in enumerate(loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data, task_id)
                    loss += self.losses[task_id](output, target).sum()
                    metric += self.metrics[task_id](output, target)
                    pbar.update()
                eval_losses[task_id] = loss.item() / len(loader.dataset)
                eval_metrics[task_id] = metric.item() / len(loader.dataset)
        pbar.close()
        return eval_losses, eval_metrics

    def train_epoch(self):
        raise NotImplementedError

    def run_epoch(self, epoch):
        log_utils.print_on_epoch_begin(epoch)

        train_losses, train_metrics = self.train_epoch()
        eval_losses, eval_metrics = self.eval_epoch()

        log_utils.print_eval_info(
                train_losses, train_metrics,
                eval_losses, eval_metrics)

        for task_id in self.task_ids:
            self.tensorboard_writer.add_scalar(
                    '{}/train/loss'.format(task_id),
                    train_losses[task_id], epoch)
            self.tensorboard_writer.add_scalar(
                    '{}/train/metric'.format(task_id),
                    train_metrics[task_id], epoch)
            self.tensorboard_writer.add_scalar(
                    '{}/val/loss'.format(task_id),
                    eval_losses[task_id], epoch)
            self.tensorboard_writer.add_scalar(
                    '{}/val/metric'.format(task_id),
                    eval_metrics[task_id], epoch)

        return eval_losses, eval_metrics
