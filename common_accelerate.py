"""
Common routines for pytorch
"""
import sys
import os
import signal
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from PyCmpltrtok.common import sep, get_path_from_prefix
from PyCmpltrtok.common_torch import CommonTorchException
import torch.distributed as dist
from torch.optim import Optimizer
import accelerate as acc


def acc_compile(
    ts=None, model=None, criterion=None,
    optim=None, ALPHA=None, GAMMA=None, GAMMA_STEP=1, GAMMA_STRATEGY: str='epoch', GAMMA_VERBOSE=True,
    warmup=0,
    metrics=None,
    ext='.pth',
    acc_kwargs={},
):
    """
    Simulate TensorFlow2's  keras.Model.compile

    :param ts: TVTS object.
    :param model: The modle.
    :param criterion: The loss.
    :param optim: The optimizer.
    :param ALPHA: The learning rate.
    :param GAMMA: The multiplicative factor of learning rate decay per GAMMA_STEP.
    :param GAMMA_STEP: How many epochs is a gamma step.
    :param metrics: The metric. Note: Now, only ONE metric is supported.
    :param ext: The extension file name of saved weights.
    :return: dict
    """
    accelerator = acc.Accelerator(**acc_kwargs)
    model_dict = {}
    model_dict['acc'] = accelerator
    model_dict['tvts'] = ts
    model_dict['device'] = accelerator.device
    
    model_dict['criterion'] = criterion
    if metrics is None:
        model_dict['metric'] = {}
    else:
        ex = CommonTorchException('If you wish to specify "metrics", it should be a name-callable dict.')
        if type(metrics) != dict:
            raise ex
        for k, v in metrics.items():
            if type(k) != str or not callable(v):
                raise ex
        model_dict['metric'] = metrics
        
    assert isinstance(ALPHA, float)
    assert ALPHA > 0.0
    model_dict['lr'] = ALPHA
        
    if optim is None:
        raise Exception('optim cannot be None!')
    elif isinstance(optim, Optimizer):
        optim = optim
    else:
        optim = optim(params=model.parameters(), lr=ALPHA)

    assert isinstance(warmup, int)
    assert warmup >= 0
    model_dict['warmup'] = warmup

    if GAMMA is None:
        lr = None
    else:
        assert GAMMA_STRATEGY in set(['step', 'epoch']), 'GAMMA_STRATEGY must be "step" or "epoch".'
        model_dict['gamma_strategy'] = GAMMA_STRATEGY
        model_dict['gamma_step'] = GAMMA_STEP
        lr = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=GAMMA, verbose=GAMMA_VERBOSE)
        if model_dict['warmup'] > 0:
            for group in lr.optimizer.param_groups:
                group['lr'] = 0.0
            print(f"**** Warning: warmup for {model_dict['warmup']} {model_dict['gamma_strategy']}(s)")
        else:
            print('**** No warmup.')
            
    model = accelerator.prepare(model)
    if optim is not None:
        optim = accelerator.prepare(optim)
    if lr is not None:
        lr = accelerator.prepare(lr)
            
    model_dict['model'] = model
    model_dict['o_model'] = accelerator.unwrap_model(model)
    model_dict['optim'] = optim
    model_dict['lr_scheduler'] = lr
                
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if world_size <= 1:
        rank = -1
                
    model_dict['ext'] = ext
    model_dict['rank'] = rank
    model_dict['global_step'] = 0
    return model_dict


def acc_process_data(model_dict, dl, is_train, label, epoch=0, world_size=1):
    """
    The work horse of training / validating / testing

    :param model_dict: Result of functon compile.
    :param dl: Dataloader.
    :param is_train: If this process is a training.
    :param label: The label such as "train", "val", "test". It will be utilized in the future.
    :param epoch: epoch number, from 1 to n.
    :return: tuple (avg_loss, avg_metric)
    """
    sep(label, cnt=16)
    rank = model_dict['rank']
    ts = model_dict['tvts']
    device = model_dict['device']
    model = model_dict['model']
    optim = model_dict['optim']
    criterion = model_dict['criterion']
    metrics_dict = model_dict['metric']
    accelerator = model_dict['acc']
    dl = accelerator.prepare(dl)
    
    avg_loss = 0.
    avg_metrics = {}
    for k in metrics_dict.keys():
        avg_metrics[k] = 0.
    for i, batch_data in enumerate(dl):
        batch = i + 1
        print('>', end='', flush=True)
        
        if isinstance(batch_data, dict):
            bx = batch_data['feature']
            by = batch_data['label']
        else:
            bx, by = batch_data
        bx = bx.float()
        by = by.long()
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            by = by.view(-1)
        if is_train:
            model_dict['global_step'] += 1
            model.train(True)
            optim.zero_grad()
            h = model(bx)

            loss = criterion(h, by)
            accelerator.backward(loss)
            optim.step()
            model.train(False)
        else:
            if rank != -1:
                o_model = model.module
            else:
                o_model = model
            o_model.train(False)
            with torch.no_grad():
                h = o_model(bx)
            if criterion is not None:
                loss = criterion(h, by)
            else:
                loss = 0
        metrics = {}
        for k, metric_routine in metrics_dict.items():
            metrics[k] = metric_routine(by, h).detach().cpu().numpy().item()

        if loss == 0:
            lossv = 0.0
        else:
            lossv = loss.detach().cpu().numpy().item()
        if is_train:
            lr = optim.param_groups[0]['lr']
            xparams = {
                'loss': lossv,
                'lr': lr,
            }
            for k, v in metrics.items():
                xparams[k] = v
            if rank in set([0, -1]):
                ts.save_batch(epoch, batch, xparams)

        avg_loss += lossv
        for k in metrics_dict.keys():
            avg_metrics[k] += metrics[k]
            
        if is_train and model_dict['gamma_strategy'] == 'step' and model_dict['lr_scheduler'] is not None:
            if model_dict['warmup'] > 0 and model_dict['global_step'] <= model_dict['warmup']:
                for group in model_dict['lr_scheduler'].scheduler.optimizer.param_groups:
                    group['lr'] = (model_dict['global_step'] / model_dict['warmup'] * model_dict['lr'])
            elif (model_dict['global_step'] - model_dict['warmup']) % model_dict['gamma_step'] == 0:
                model_dict['lr_scheduler'].step()
    print()
    avg_loss /= i + 1
    for k in metrics_dict.keys():
        avg_metrics[k] /= i + 1
    return avg_loss, avg_metrics


def acc_fit(model_dict, dl_train, dl_val, n_epochs, world_size=1):
    """
    Simulate TensorFlow2's  keras.Model.fit

    :param model_dict: Result of functon compile.
    :param dl_train: DataLoader of training data.
    :param dl_val: DataLoader of validating data.
    :param n_epochs: How many epochs to train.
    :return: None
    """
    rank = model_dict['rank']
    ts = model_dict['tvts']
    model = model_dict['model']
    optim = model_dict['optim']
    ext = model_dict['ext']

    def save_model(save_name):
        save_prefix = os.path.join(ts.save_dir, save_name)
        save_path = get_path_from_prefix(save_prefix, ext)
        os.makedirs(ts.save_dir, exist_ok=True)
        print(f'Saving to {save_path}')
        sdict = model.state_dict()
        torch.save(sdict, save_path)
        print('Saved')
        return save_path

    epoch = 1

    def signal_handler(sig, frame):
        save_model(f'{ts.name}-{ts.train_id}-{epoch}-interrupted')
        sys.exit(0)

    if rank in set([0, -1]):
        signal.signal(signal.SIGINT, signal_handler)

    if rank in set([0, -1]):
        ts.mark_start_dt()
    for idx in range(n_epochs):
        epoch = idx + 1
        sep(epoch)
        avg_loss, avg_metrics = acc_process_data(model_dict, dl_train, True, 'train', epoch, world_size)
        if rank in set([0, -1]):
            avg_loss_val, avg_metrics_val = acc_process_data(model_dict, dl_val, False, 'val', epoch, world_size)
        if rank in set([0, -1]):
            print(f'epoch#{epoch + 1}: loss = {avg_loss} metrics = {avg_metrics}, loss_val = {avg_loss_val}, metrics_val = {avg_metrics_val}')
        else:
            print(f'epoch#{epoch + 1}: loss = {avg_loss} metrics = {avg_metrics}')
        if rank in set([0, -1]):
            if epoch % ts.save_freq == 0 or epoch == n_epochs:
                save_name = ts.get_save_name(epoch)
                save_path = save_model(save_name)
                save_rel_path = os.path.relpath(save_path, ts.save_dir)
            else:
                save_rel_path = None

            lr = optim.param_groups[0]['lr']
            xparams = {
                'loss': avg_loss,
                'loss_val': avg_loss_val,
                'lr': lr,
            }
            for k, v in avg_metrics.items():
                xparams[k] = v
            for k, v in avg_metrics_val.items():
                xparams[k+'_val'] = v
            ts.save_epoch(epoch, xparams, save_rel_path, ts.save_dir)

        if model_dict['gamma_strategy'] == 'epoch' and model_dict['lr_scheduler'] is not None:
            if model_dict['warmup'] > 0 and epoch <= model_dict['warmup']:
                for group in model_dict['lr_scheduler'].optimizer.param_groups:
                    group['lr'] = (epoch / model_dict['warmup'] * model_dict['lr'])
            elif (epoch - model_dict['warmup']) % model_dict['gamma_step'] == 0:
                model_dict['lr_scheduler'].step()


def acc_evaluate(model_dict, dl_test):
    avg_loss_test, avg_metrics_test = acc_process_data(model_dict, dl_test, False, 'test')
    return avg_loss_test, avg_metrics_test
