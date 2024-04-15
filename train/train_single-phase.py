import argparse
import math
import os

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import XrdDataset
from model.CPICANN import CPICANN
from model.focal_loss import FocalLoss
from util.logger import Logger


def get_acc(cls, label):
    cls_acc = sum(cls.argmax(1) == label.int()) / cls.shape[0]
    return cls_acc


def run_one_epoch(model, dataloader, criterion, optimizer, epoch, mode):
    if mode == 'Train':
        model.train()
        criterion.train()
        desc = 'Training... '
    else:
        model.eval()
        criterion.eval()
        desc = 'Evaluating... '

    epoch_loss, cls_acc = 0, 0
    if args.progress_bar:
        pbar = tqdm(total=len(dataloader.dataset), desc=desc, unit='data')
    iters = len(dataloader)
    for i, batch in enumerate(dataloader):
        data = batch[0].to(device)
        label_cls = batch[1].to(device)

        if mode == 'Train':
            adjust_learning_rate_withWarmup(optimizer, epoch + i / iters, args)

            logits = model(data)
            loss = criterion(logits, label_cls.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(data)
                loss = criterion(logits, label_cls.long())

        epoch_loss += loss.item()
        if args.progress_bar:
            pbar.update(len(data))
            pbar.set_postfix(**{'loss': loss.item()})

        _cls_acc = get_acc(logits, label_cls)
        cls_acc += _cls_acc.item()

    return epoch_loss / iters, cls_acc * 100 / iters


def print_log(epoch, loss_train, loss_val, acc_train, acc_val, lr):
    log.printlog('---------------- Epoch {} ----------------'.format(epoch))

    log.printlog('loss_train : {}'.format(round(loss_train, 4)))
    log.printlog('loss_val   : {}'.format(round(loss_val, 4)))

    log.printlog('acc_train  : {}%'.format(round(acc_train, 4)))
    log.printlog('acc_val    : {}%'.format(round(acc_val, 4)))

    log.train_writer.add_scalar('loss', loss_train, epoch)
    log.val_writer.add_scalar('loss', loss_val, epoch)

    log.train_writer.add_scalar('acc', acc_train, epoch)
    log.val_writer.add_scalar('acc', acc_val, epoch)

    log.train_writer.add_scalar('lr', lr, epoch)


def save_checkpoint(state, is_best, filepath, filename):
    if (state['epoch']) % 10 == 0 or state['epoch'] == 1:
        os.makedirs(filepath, exist_ok=True)
        torch.save(state, filepath + filename)
        log.printlog('checkpoint saved!')
        if is_best:
            torch.save(state, '{}/model_best.pth'.format(filepath))
            log.printlog('best model saved!')


def adjust_learning_rate_withWarmup(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    print('>>>>  Running on {}  <<<<'.format(device))

    model = CPICANN(embed_dim=128, num_classes=args.num_classes)
    model.to(device)
    if rank == 0:
        log.printlog(model)

    trainset = XrdDataset(args.data_dir_train, args.anno_train)
    valset = XrdDataset(args.data_dir_val, args.anno_val)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=True)

        train_loader = DataLoader(trainset, batch_size=128, num_workers=16, pin_memory=True, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(valset, batch_size=128, num_workers=16, pin_memory=True, drop_last=True, sampler=val_sampler)

        model = DDP(model, device_ids=[device], output_device=local_rank, find_unused_parameters=False)
    else:
        train_loader = DataLoader(trainset, batch_size=128, num_workers=16, pin_memory=True, shuffle=True)
        val_loader = DataLoader(valset, batch_size=128, num_workers=16, pin_memory=True, shuffle=True)

    criterion = FocalLoss(class_num=args.num_classes, device=device)

    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    start_epoch = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        loss_train, acc_train = run_one_epoch(model, train_loader, criterion, optimizer, epoch, mode='Train')

        loss_val, acc_val = run_one_epoch(model, val_loader, criterion, optimizer, epoch, mode='Eval')

        if rank == 0:
            print_log(epoch,  loss_train, loss_val, acc_train, acc_val, optimizer.param_groups[0]['lr'])
            save_checkpoint({'epoch': epoch,
                             'model': model.module.state_dict() if distributed else model.state_dict(),
                             'optimizer': optimizer}, is_best=False,
                            filepath='{}/checkpoints/'.format(log.get_path()),
                            filename='checkpoint_{:04d}.pth'.format(epoch))


if __name__ == '__main__':
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        distributed = True
    else:
        rank = 0
        device = 'cuda:0'
        distributed = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_bar", type=bool, default=True)

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup-epochs', default=20, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')

    parser.add_argument('--data_dir_train', default='data/train/', type=str)
    parser.add_argument('--data_dir_val', default='data/val/', type=str)
    parser.add_argument('--anno_train', default='annotation/anno_train.csv', type=str,
                        help='path to annotation file for training data')
    parser.add_argument('--anno_val', default='annotation/anno_val.csv', type=str,
                        help='path to annotation file for validation data')
    parser.add_argument('--num_classes', default=23073, type=int, metavar='N')

    args = parser.parse_args()

    if rank == 0:
        log = Logger(val=True)

    main()
    print('THE END')
