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
import torch.distributed as dist


class CommonTorchException(Exception):
    pass


def torch_compile(
    ts=None, device=None, model=None, criterion=None,
    optim=None, ALPHA=None, GAMMA=None, GAMMA_STEP=1,
    metrics=None,
    ext='.pth',
    rank=-1,
):
    """
    Simulate TensorFlow2's  keras.Model.compile

    :param ts: TVTS object.
    :param device: The device to use, GPU or CPU.
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
    model_dict = {}
    model_dict['tvts'] = ts
    model_dict['device'] = device
    model_dict['model'] = model
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
    if optim is None:
        model_dict['optim'] = None
    else:
        model_dict['optim'] = optim(params=model.parameters(), lr=ALPHA)

    if GAMMA is None or np.isclose(1.0, GAMMA):
        model_dict['lr_scheduler'] = None
    else:
        model_dict['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(model_dict['optim'], step_size=GAMMA_STEP, gamma=GAMMA, verbose=True)
    model_dict['ext'] = ext
    model_dict['rank'] = rank
    return model_dict


def torch_acc_top1(y, pred):
    """
    Top-1 accuracy.

    :param y: The groud truth vector.
    :param pred: The prediction tensor in shape (batch, n_cls)
    :return: the top-1 accuracy value.
    """
    y = y.long()
    pred = pred.argmax(dim=1)
    acc = torch.eq(y, pred).float().mean()
    return acc


def torch_acc_topn(y, pred, n):
    """
    Top-1 accuracy.

    :param y: The groud truth vector.
    :param pred: The prediction tensor in shape (batch, n_cls)
    :return: the top-1 accuracy value.
    """
    n_cls = pred.shape[1]
    assert n <= n_cls
    y = y.long().reshape(-1, 1)
    pred = pred.argsort(dim=1)[:, n_cls-n:n_cls]
    ys = torch.tile(y, (n,))
    acc = torch.any(torch.eq(ys, pred), dim=1).float().mean()
    return acc


def torch_acc_top2(y, pred):
    return torch_acc_topn(y, pred, 2)


def torch_process_data(model_dict, dl, is_train, label, epoch=0, world_size=1):
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
    avg_loss = 0.
    avg_metrics = {}
    for k in metrics_dict.keys():
        avg_metrics[k] = 0.
    for i, (bx, by) in enumerate(dl):
        batch = i + 1
        print('>', end='', flush=True)
        bx = bx.float().to(device)
        by = by.long().to(device)
        if is_train:
            model.train(True)
            optim.zero_grad()
            h = model(bx)
            loss = criterion(h, by)
            loss.backward()
            optim.step()
            model.train(False)
        else:
            model.train(False)
            h = model(bx)
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
    print()
    avg_loss /= i + 1
    for k in metrics_dict.keys():
        avg_metrics[k] /= i + 1
    return avg_loss, avg_metrics


def torch_fit(model_dict, dl_train, dl_val, n_epochs, world_size=1):
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
    for epoch in range(n_epochs):
        epoch += 1
        sep(epoch)
        avg_loss, avg_metrics = torch_process_data(model_dict, dl_train, True, 'train', epoch, world_size)
        avg_loss_val, avg_metrics_val = torch_process_data(model_dict, dl_val, False, 'val', epoch, world_size)
        print(f'epoch#{epoch + 1}: loss = {avg_loss} acc = {avg_metrics}, loss_val = {avg_loss_val}, acc_val = {avg_metrics_val}')

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

        if model_dict['lr_scheduler'] is not None:
            model_dict['lr_scheduler'].step()


def torch_evaluate(model_dict, dl_test):
    avg_loss_test, avg_metrics_test = torch_process_data(model_dict, dl_test, False, 'test')
    return avg_loss_test, avg_metrics_test


def torch_infer(x, model, device, batch_size):
    """
    Infer

    :param x: Input tensor as (N, C, H, W)
    :param model: The model.
    :param device: The device, GPU or CPU.
    :param batch_size: Batch size used when inferring.
    :return:
    """
    x = torch.Tensor(x)
    ds_x = TensorDataset(x)
    dl_x = DataLoader(ds_x, batch_size, drop_last=False)

    net = model
    net.eval()
    pred = None
    for bx, in dl_x:
        bx = bx.to(device)
        bpred = net(bx)
        bpred = bpred.detach().cpu().numpy()
        if pred is None:
            pred = bpred
        else:
            pred = np.concatenate([pred, bpred], axis=0)
    return pred


def patchify_fast(images, nh, nw=None):
    if nw is None:
        nw = nh
    n, c, h, w = images.shape

    assert h % nh == 0, "Input shape not entirely divisible by number of patches"
    assert w % nw == 0, "Input shape not entirely divisible by number of patches"
    hsize = h // nh
    wsize = w // nw

    patches = images
    # print(patches.shape)  # n, c, h, w
    if isinstance(patches, np.ndarray):
        is_np = True
    else:
        is_np = False

    trans_tuple = (0, 2, 3, 1)  # n, h, w, c
    if is_np:
        patches = np.transpose(patches, trans_tuple)
    else:
        patches = torch.permute(patches, trans_tuple)
    # print(patches.shape)
    patches = patches.reshape(n, nh, hsize, nw, wsize, c)  # as code
    # print(patches.shape)

    trans_tuple = (0, 1, 3, 5, 2, 4)  # n, nh, nw, c, hsize, wsize
    if is_np:
        patches = np.transpose(patches, trans_tuple)
    else:
        patches = torch.permute(patches, trans_tuple)
    # print(patches.shape)
    patches = patches.reshape(n, nh * nw, c * hsize * wsize)
    # print(patches.shape)

    return patches
