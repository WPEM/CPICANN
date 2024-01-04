import argparse
import csv
import os

import pandas as pd
import torch
from torch.nn.functional import softmax
import numpy as np

import plot
import data_format as formater
from model import DPID


def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=str, default='cuda:0',
                        help="device to run the model, example: 'cuda:0', 'cpu'. ")
    parser.add_argument("-data_path", type=str, default='data/single-phase')
    parser.add_argument("-inf_mode", type=str, default='single-phase',
                        help="single-phase, di-phase")
    parser.add_argument("-include_elements_must", type=str, default='Fe',
                        help="elements to be included in the prediction, example: 'Fe'. ")
    parser.add_argument("-include_elements_atLeastOne", type=str, default='O_C_S',
                        help="elements to be included in the prediction, example: 'O_C_S'. ")
    parser.add_argument("-exclude_elements", type=str, default='',
                        help="elements to be excluded in the prediction, example: 'Fe_O'. ")
    args = parser.parse_args()

    infDir = args.data_path

    if not os.path.exists('figs'):
        os.makedirs('figs')

    # get annotation data for display purposeys = np.zeros(4500)
    annoMap, elemMap = get_anno_map()

    # get specific element setting map
    elem_setting_map = get_elem_setting()

    # choose the model to be loaded
    if args.inf_mode == 'single-phase':
        load_path = 'pretrained/DPID_single-phase.pth'
    elif args.inf_mode == 'di-phase':
        load_path = 'pretrained/DPID_di-phase.pth'
    else:
        raise ValueError('invalid inf_mode: {}'.format(args.inf_mode))

    # load model parameters
    model = DPID(embed_dim=128, num_classes=23073)
    loaded = torch.load(load_path)
    model.load_state_dict(loaded['model'])
    print('loaded model from {}'.format(load_path))
    if args.device != 'cpu':
        model.to(args.device)
    model.eval()

    # write the inference results to a csv file
    resFileName = 'infResults_{}.csv'.format(infDir.split('/')[-1])
    dataFile = open(resFileName, 'w')
    dataWriter = csv.writer(dataFile)
    dataWriter.writerow(['path', 'fileName', 'predRank', 'pred', 'codId', 'formula', 'spaceGroupNo', 'spaceGroup'])

    fileList = os.listdir(infDir)
    fileList.sort()
    for l in fileList:
        filePath = os.path.join(infDir, l)
        print('\n>>>>>> RUNNING: {}'.format(filePath))

        data = formater.convert_file(filePath)
        if data is None:
            continue

        logits = inference(model, data, args).squeeze()

        try:
            elem_setting = elem_setting_map[l]

            # use specific element setting
            logits = filter_elem(logits, elemMap, elem_setting)
        except KeyError:
            # use global element setting
            # filter by include elements
            logits = filter_elem(logits, elemMap,
                                 {'include_must': args.include_elements_must,
                                  'include_atLeastOne': args.include_elements_atLeastOne,
                                  'exclude': args.exclude_elements})

        predList = []
        if args.inf_mode == 'single-phase':
            pred = logits.argmax().item()
            info = annoMap[pred]
            predList.append(info)
            conf = torch.nn.functional.softmax(logits, dim=0).max().item()

            print('pred cls_id : {}  confidence : {:.2f}%'.format(pred, conf * 100))
            print('pred cod_id : {}  formula : {}'.format(int(info[0]), info[2]))
            print('pred space group No: {}    space group : {}'.format(info[5], info[4]))
            dataWriter.writerow([infDir, l, 1, pred, info[0], info[2], info[5], info[4]])
        else:
            logits = torch.nn.functional.softmax(logits, dim=0)
            top5 = logits.topk(5)

            for i, (pred, conf) in enumerate(zip(top5[1], top5[0])):
                if i >= 2 and conf < 0.1:
                    break
                info = annoMap[pred.item()]
                predList.append(info)
                print('>>> pred rank {}'.format(i+1))
                print('pred cls_id : {}  confidence : {:.4f}%'.format(pred.item(), conf * 100))
                print('pred cod_id : {}  formula : {}'.format(int(info[0]), info[2]))
                print('pred space group No: {}    space group : {}'.format(info[5], info[4]))
                dataWriter.writerow([infDir, l, i+1, pred.item(), info[0], info[2], info[5], info[4]])

        plot.run(l, data, predList)

    dataFile.close()

    return resFileName


def inference(model, v, args):
    v = v / v.max() * 100

    v = torch.tensor(v, dtype=torch.float32).reshape(1, 1, -1)
    if args.device != 'cpu':
        v = v.to(args.device)
    with torch.no_grad():
        logits = model(v)
    return logits


def filter_elem(logits, elemMap, elem_setting):
    include_must = elem_setting['include_must']
    include_atLeastOne = elem_setting['include_atLeastOne']
    exclude = elem_setting['exclude']

    if include_must != "":
        exclude_index = get_index_by_include(elemMap, include_must)
        logits = filter_by_index(logits, exclude_index)

    if include_atLeastOne != "":
        exclude_index = get_index_by_include_atLeastOne(elemMap, include_atLeastOne)
        logits = filter_by_index(logits, exclude_index)

    if exclude != "":
        exclude_index = get_index_by_exclude(elemMap, exclude)
        logits = filter_by_index(logits, exclude_index)

    return logits


def get_anno_map():
    vs = pd.read_csv('config/strucs.csv').values
    annos = {}
    elems = pd.DataFrame({'No': vs[:, 1], 'elem': vs[:, 3]})
    for v in vs:
        annos[v[1]] = v
    return annos, elems


def get_elem_setting():
    try:
        vs = pd.read_csv('config/elem_setting.csv').values
        elem_setting_map = {}
        for v in vs:
            include_must = "" if str(v[1]) == 'nan' else v[1]
            include_atLeastOne = "" if str(v[2]) == 'nan' else v[2]
            exclude = "" if str(v[3]) == 'nan' else v[3]
            elem_setting_map[v[0].strip()] = {'include_must': include_must,
                                              'include_atLeastOne': include_atLeastOne,
                                              'exclude': exclude}
    except FileNotFoundError:
        elem_setting_map = {}
    return elem_setting_map


def get_index_by_include(elemMap, include):
    include = include.split('_')
    if len(include) == 0:
        return []

    excludeIndex = [0] * len(elemMap)
    for i, elems in elemMap.values:
        for e in include:
            if e not in set(elems.split(' ')):
                excludeIndex[i] = 1
                break

    excludeIndex = np.array(excludeIndex).nonzero()[0]
    return excludeIndex


def get_index_by_include_atLeastOne(elemMap, include):
    include = include.split('_')
    if len(include) == 0:
        return []

    excludeIndex = [1] * len(elemMap)
    for i, elems in elemMap.values:
        for e in include:
            if e in set(elems.split(' ')):
                excludeIndex[i] = 0
                break

    excludeIndex = np.array(excludeIndex).nonzero()[0]
    return excludeIndex


def get_index_by_exclude(elemMap, exclude):
    exclude = set(exclude.split('_'))
    if len(exclude) == 0:
        return []

    excludeIndex = [0] * len(elemMap)
    for i, es in elemMap.values:
        for e in es.split(' '):
            if e in exclude:
                excludeIndex[i] = 1
                break

    excludeIndex = np.array(excludeIndex).nonzero()[0]
    return excludeIndex


def filter_by_index(logits, index):
    if len(index) == 0:
        return logits

    logits[index] = -1e10
    return logits


if __name__ == '__main__':
    resFileName = main()
    print('\ninference result saved in {}'.format(resFileName))
    print('inference figures saved at figs/')
    print('THE END')
