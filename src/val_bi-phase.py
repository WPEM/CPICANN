import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.CPICANN import CPICANN


def getAnnoMap():
    vs = pd.read_csv(args.anno_struc).values
    annos, elems = {}, {}
    for v in vs:
        annos[v[1]] = v
        elems[v[1]] = set(v[3].split(' '))

    return annos, elems


def filter_by_elem(logits, elemMap, elem):
    for i, e in elemMap.items():
        if not e <= elem:
            logits[:, i] = -10 ** 9

    return logits


def main():
    annoMap, elemMap = getAnnoMap()

    model = CPICANN(embed_dim=128, num_classes=args.num_classes)

    loaded = torch.load(args.load_path)
    model.load_state_dict(loaded['model'])
    model.to(args.device)
    model.eval()
    print('loaded model from {}'.format(args.load_path))
    print(model)

    if args.elem_filtration:
        print('elem_filtration activated!')
    else:
        print('elem_filtration deactivated!')

    lst = pd.read_csv(args.anno_val).values

    top10Hits = np.array([0] * 10, dtype=np.int32)

    dataLen = len(lst)
    pbar = tqdm(range(args.infTimes))
    for i in range(args.infTimes):
        while True:
            c1, c2 = np.random.randint(0, dataLen, 2)
            anno1, anno2 = lst[c1], lst[c2]
            if anno1[6] != anno2[6]:
                break

        # id1, id2 = int(lst[c1][0].split('_')[0]), int(lst[c2][0].split('_')[0])
        # formula1, formula2 = lst[c1][2], lst[c2][2]
        data1 = pd.read_csv(os.path.join(args.data_dir, f'{lst[c1][0]}.csv')).values
        data2 = pd.read_csv(os.path.join(args.data_dir, f'{lst[c2][0]}.csv')).values

        mixRate1 = np.random.randint(20, 81)
        mixRate2 = 100 - mixRate1

        data = mixRate1 * data1 + mixRate2 * data2
        elem = set(lst[c2][3].strip().split(' ')) | set(lst[c1][3].strip().split(' '))

        def runFile(v):
            min_i, scale = min(v), max(v) - min(v)
            v = (v - min_i) / scale * 100

            v = torch.tensor(v, dtype=torch.float32).reshape(1, 1, -1)
            v = v.to(args.device)
            with torch.no_grad():
                logits = model(v)

                # filter by elements
                if args.elem_filtration:
                    logits = filter_by_elem(logits, elemMap, elem)

                _pred = torch.nn.functional.softmax(logits.squeeze(), dim=0)
            return _pred.topk(10)

        top10 = runFile(data)

        m = [0] * 10
        for no, (indice, rate) in enumerate(zip(top10.indices, top10.values)):
            pred = annoMap[top10.indices[no].item()]

            if pred[0] == int(anno1[0][:7]):
                m[no] = 1
            elif pred[0] == int(anno2[0][:7]):
                m[no] = 2

        if 1 in m[:2] and 2 in m[:2]:
            top10Hits[1:] += 1
        elif 1 in m[:3] and 2 in m[:3]:
            top10Hits[2:] += 1
        elif 1 in m[:4] and 2 in m[:4]:
            top10Hits[3:] += 1
        elif 1 in m[:5] and 2 in m[:5]:
            top10Hits[4:] += 1
        elif 1 in m[:6] and 2 in m[:6]:
            top10Hits[5:] += 1
        elif 1 in m[:7] and 2 in m[:7]:
            top10Hits[6:] += 1
        elif 1 in m[:8] and 2 in m[:8]:
            top10Hits[7:] += 1
        elif 1 in m[:9] and 2 in m[:9]:
            top10Hits[8:] += 1
        elif 1 in m[:10] and 2 in m[:10]:
            top10Hits[9:] += 1

        pbar.update(1)
    pbar.close()

    for i in range(1, 10):
        print('top{}Hits: {}%'.format(i + 1, round(top10Hits[i] / args.infTimes * 100, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--data_dir', default='data/val/', type=str)
    parser.add_argument('--infTimes', default=1000, type=int, help='number of mixed pattern to be inferenced')
    parser.add_argument('--load_path', default='pretrained/bi-phase_checkpoint_2000.pth', type=str,
                        help='path to load pretrained single-phase identification model')
    parser.add_argument('--anno_struc', default='annotation/anno_struc.csv', type=str,
                        help='path to annotation file for training data')
    parser.add_argument('--anno_val', default='annotation/anno_val.csv', type=str,
                        help='path to annotation file for validation data')
    parser.add_argument('--num_classes', default=23073, type=int, metavar='N')

    parser.add_argument('--elem_filtration', default=False, type=bool)

    args = parser.parse_args()

    main()
    print('THE END')
