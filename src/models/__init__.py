import os
import sys
import pandas as pd
import torch

from .lenet import Lenet


class ModelManager:
    """Manages model checkpoints, saving and loading.
    """
    def __init__(self, checkpoint_dir, task_ids):
        self.checkpoint_dir = os.path.expanduser(checkpoint_dir)
        self.task_ids = task_ids
        if os.path.isfile(os.path.join(checkpoint_dir, 'history.csv')):
            self.read_history(os.path.join(checkpoint_dir, 'history.csv'))
        else:
            self.history = dict()
            self.last_epoch = -1

    def read_history(self, history_path):
        """Converts a saved `*.csv` to a list of losses at each epoch
        """
        df = pd.read_csv(os.path.expanduser(history_path), index_col=0)
        if not set(self.task_ids).issubset(set(df.columns.values.tolist())):
            raise ValueError
        self.history = df.to_dict('index')
        self.last_epoch = max(self.history.keys())

    def load_model(self, model_name, model_weights, model_kwargs):
        """Dynamically loads the specified `nn.Module` object
        """
        if not hasattr(sys.modules[__name__], model_name):
            raise ValueError
        model_def = getattr(sys.modules[__name__], model_name)
        model = model_def(**model_kwargs)
        last_model = None
        if model_weights is not None:
            last_model = os.path.expanduser(model_weights)
            model.load_state_dict(torch.load(last_model))
        elif len(self.history) > 0:
            last_model = os.path.expanduser(os.path.join(
                    self.checkpoint_dir,
                    self.history[self.last_epoch]['checkpoint']))
            model.load_state_dict(torch.load(last_model))
        return model, last_model

    def save_model(self, model, losses, epoch=None):
        """Saves the model checkpoint, and dumps the losses to `history.csv`
        """
        if not set(self.task_ids) == set(losses.keys()):
            raise ValueError
        epoch_n = self.last_epoch + 1 if epoch is None else epoch
        if not epoch_n > self.last_epoch:
            raise ValueError('Current epoch ({}) must be larger than last '
                             'epoch ({}).'.format(epoch_n, self.last_epoch))
        checkpoint_name = '{}.pth'.format(epoch_n)
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        self.history[epoch_n] = dict([
            (task_id, losses[task_id]) for task_id in self.task_ids])
        self.history[epoch_n]['checkpoint'] = checkpoint_name
        torch.save(model.state_dict(), checkpoint_path)
        self.dump_history()

    def dump_history(self):
        """Dumps the history into `checkpoint_dir/history.csv` file.
        """
        df = pd.DataFrame.from_dict(self.history, orient='index')
        df.index.name = 'epoch'
        df.to_csv(os.path.join(self.checkpoint_dir, 'history.csv'))
