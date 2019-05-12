from tqdm import tqdm
import torch

from utils import log_utils


class BaseTrainer:
    """
    Base class for training scripts. Just a convenient wrapper, with
    evaluation and early stopping functions already implemented.

    Attributes:
      device:               either 'cuda' or 'cpu'
      model:                instance of a class, inherited from `Hydra`
      model_manager:        instance of a `ModelManager` class
      task_ids:             list of unique task ids
      losses:               dictionary of losses {task_id: loss_function}
      metrics:              dictionary of metrics {task_id: metric}
      train_loaders:        an instance of `torch.utils.data.DataLoader`
      test_loaders:         an instance of `torch.utils.data.DataLoader`
      tensorboard_writer:   an instance of `tensorboardX.SummaryWriter`
    """

    def __init__(self,
                 device,
                 model,
                 losses,
                 metrics,
                 train_loaders,
                 test_loaders=None,
                 model_manager=None,
                 tensorboard_writer=None,
                 patience=None,
                 **kwargs):

        self.device = device
        self.model = model
        self.model_manager = model_manager
        self.losses = losses
        self.metrics = metrics
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.tensorboard_writer = tensorboard_writer

        self.patience = patience
        self.best_score = None
        self.counter = 0

        assert set(self.model.heads.keys()) <= set(losses.keys())
        assert set(self.model.heads.keys()) <= set(metrics.keys())
        assert set(self.model.heads.keys()) <= set(train_loaders.keys())
        if test_loaders is not None:
            assert set(self.model.heads.keys()) <= set(test_loaders.keys())

        self.task_ids = list(self.model.heads.keys())

    def eval_epoch(self):
        """Evaluates `self.model` for one epoch over test `self.test_loaders`
        """
        if self.test_loaders is None:
            raise ValueError('test_loaders not specified, cannot evaluate')

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
        """Trains `self.model` for one epoch over test `self.test_loaders`
        """
        raise NotImplementedError

    def run_epoch(self, epoch):
        """Runs `train_epoch()` and `test_epoch()` for one epoch
        """
        log_utils.print_on_epoch_begin(epoch, self.counter)

        train_losses, train_metrics = self.train_epoch()
        if self.test_loaders is not None:
            eval_losses, eval_metrics = self.eval_epoch()

            log_utils.print_eval_info(
                    train_losses, train_metrics,
                    eval_losses, eval_metrics)

        if self.tensorboard_writer is not None:
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

        if self.patience is not None:
            if self.best_score is None:
                self.best_score = eval_losses

            has_improved = True
            for task_id in self.task_ids:
                has_improved = has_improved and \
                    (self.best_score[task_id] + 1e-5 >= eval_losses[task_id])
            if has_improved:
                self.best_score = eval_losses
                self.counter = 0
            else:
                self.counter += 1
        return eval_losses, eval_metrics

    def early_stop(self):
        """Returns `True` if `self.model` hasn't improved for a long time
        """
        if self.patience is None:
            return False
        return self.counter > self.patience
