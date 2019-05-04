from tqdm import tqdm
import torch


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
                    output = self.model(data, task_id=task_id)
                    loss += self.losses[task_id](output, target).sum()
                    metric += self.metrics[task_id](output, target)
                    pbar.update()
                eval_losses[task_id] = loss.item() / len(loader.dataset)
                eval_metrics[task_id] = metric.item() / len(loader.dataset)
        pbar.close()
        return eval_losses, eval_metrics

    def train_epoch(self):
        raise NotImplementedError
