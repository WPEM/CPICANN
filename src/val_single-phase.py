import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.CPICANN import CPICANN
from model.dataset import XrdDataset


def get_cs_anno():
    vs = pd.read_csv(args.anno_struc).values
    csAnno = {}
    for v in vs:
        csAnno[v[1]] = v[6]
    return csAnno


def get_acc(cls, label):
    correct_cnt = sum(cls.argmax(1) == label.int())
    cls_acc = correct_cnt / cls.shape[0]
    return cls_acc, correct_cnt


def run_one_epoch(model, dataloader):
    model.eval()

    csAnno = get_cs_anno()

    csCorrect = [0 for _ in range(7)]
    csTotal = [0 for _ in range(7)]
    cMtrx = [[0 for _ in range(7)] for _ in range(7)]
    epoch_loss, cls_acc = 0, 0
    correct_cnt, total_cnt = 0, 0
    pbar = tqdm(total=len(dataloader.dataset), desc='Evaluating... ', unit='data')
    iters = len(dataloader)
    for i, batch in enumerate(dataloader):

        data = batch[0].to(args.device)
        label_cls = batch[1].to(args.device)

        with torch.no_grad():
            logits = model(data)
            logits.to(args.device)

        pbar.update(len(data))

        _cls_acc, correct = get_acc(logits, label_cls)
        cls_acc += _cls_acc.item()

        correct_cnt += correct.item()
        total_cnt += len(data)

        preds = logits.argmax(1)
        for gt, pred in zip(label_cls, preds):
            cs_gt = csAnno[gt.item()]
            cMtrx[cs_gt][csAnno[pred.item()]] += 1
            csTotal[cs_gt] += 1
            if gt == pred:
                csCorrect[cs_gt] += 1

    return epoch_loss / iters, cls_acc * 100 / iters, correct_cnt, total_cnt, cMtrx, csCorrect, csTotal


def main():
    model = CPICANN(embed_dim=128, num_classes=args.num_classes)

    loaded = torch.load(args.load_path)
    model.load_state_dict(loaded['model'])
    model.to(args.device)
    model.eval()
    print('loaded model from {}'.format(args.load_path))

    print(model)

    valset = XrdDataset(args.data_dir, args.anno_val)
    val_loader = DataLoader(valset, batch_size=128, num_workers=16, pin_memory=True, shuffle=True)

    loss_val, acc_val, correct_cnt, total_cnt, cMtrx, csCorrect, csTotal = run_one_epoch(model, val_loader)

    print("loss_val: ", loss_val)
    print("acc_val: ", acc_val)
    print("{}%  ({}/{})".format(round(correct_cnt / total_cnt, 5) * 100, correct_cnt, total_cnt))

    sums = np.array(cMtrx).sum(axis=1)
    for i, row in enumerate(cMtrx):
        buf = ""
        for j, v in enumerate(row):
            buf += "{}({}%) ".format(v, round(v / sums[i] * 100, 2))
        print(buf)

    print("csCorrect: ", csCorrect)
    print("csTotal: ", csTotal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--data_dir', default='data/val/', type=str)
    parser.add_argument('--load_path', default='pretrained/single-phase_checkpoint_0200.pth', type=str,
                        help='path to load pretrained single-phase identification model')
    parser.add_argument('--anno_struc', default='annotation/anno_struc.csv', type=str,
                        help='path to annotation file for training data')
    parser.add_argument('--anno_val', default='annotation/anno_val.csv', type=str,
                        help='path to annotation file for validation data')
    parser.add_argument('--num_classes', default=23073, type=int, metavar='N')

    args = parser.parse_args()

    main()

    print('THE END')
