from tqdm import tqdm
from utils import log_utils

import torch
import torch.optim as optim
import torch.nn.functional as F


def train_epoch(model, device, train_loader, optimizer):
    model.train()
    print('Training:')

    pbar = tqdm(desc='  Task #1', total=len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pbar.update()
    pbar.close()


def eval_epoch(model, device, test_loader):
    model.eval()
    print('Evaluating:')
    test_loss = 0
    correct = 0
    pbar = tqdm(desc='  Task #1', total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                    output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update()
    pbar.close()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    title_format = '{:<15} | {:>8} | {:>8}'
    row_format = '{:<15} | {:8.5f} | {:8.5f}'
    between = '-' * 16 + '+' + '-' * 10 + '+' + '-' * 11

    print('EPOCH END RESULT:')
    print(between)
    print(title_format.format('Task name', 'Loss', 'Acc'))
    print(between)
    print(row_format.format('Task #1', test_loss, accuracy))
    print(between)
    losses = {'Mnist': test_loss}
    return losses


def simple(device,
           train_loader,
           test_loader,
           model,
           batch_size,
           tensorboard_dir,
           model_manager,
           epochs=1,
           optimizers=None):
    """Simple training function, assigns an optimizer for each task.
    """
    # Load Optimizers
    if optimizers is None:
        optimizer = optim.SGD(model.parameters())
    else:
        optimizer = getattr(optim, optimizers['method'])(
                model.parameters(), **optimizers['kwargs'])

    starting_epoch = model_manager.last_epoch + 1
    for epoch in range(starting_epoch, starting_epoch + epochs):
        train_epoch(model, device, train_loader, optimizer)
        losses = eval_epoch(model, device, test_loader)
        model_manager.save_model(model, losses, epoch)

