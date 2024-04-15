# It is the phase identification module of WPEM.

import csv
import os
import datetime
import pandas as pd
import torch
import numpy as np
import sys
import wget
from .plot import _run 
import zipfile
import pkg_resources
from .data_format import convert_file
from .model import CPICANN_main
from art import text2art

def PhaseIdentifier(FilePath,Task='single-phase',ElementsSystem='',ElementsContained='',ElementsExclude='',Device='cuda:0',):
    """
    CPICANN : Crystallographic Phase Identifier of Convolutional self-Attention Neural Network

    Contributors : Shouyang Zhang & Bin Cao
    ================================================================
        Please feel free to open issues in the Github :
        https://github.com/WPEM/CPICANN
        or 
        contact Mr.Bin Cao (bcao686@connect.hkust-gz.edu.cn)
        in case of any problems/comments/suggestions in using the code. 
    ==================================================================

    :param FilePath 

    :param Task, type=str, default='single-phase'
        if Task = 'single-phase', CPICANN executes a single phase identification task
        if Task = 'di-phase', CPICANN executes a dual phase identification task
    
    :param ElementsSystem, type=str, default=''
        Specifies the elements to be included at least in the prediction, example: 'Fe'.

    :param ElementsContained, type=str, default=''
        Specifies the elements to be included, with at least one of them in the prediction, example: 'O_C_S'.

    :param ElementsExclude, type=str, default=''
        Specifies the elements to be excluded in the prediction, example: 'Fe_O'

    :param Device, type=str, default='cuda:0',
        Which device to run the CPICANN, example: 'cuda:0', 'cpu'.

    examples:
    from WPEMPhase import CPICANN
    CPICANN.PhaseIdentifier(FilePath='./single-phase',Device='cpu')
    """
    
    extract_to_folder = pkg_resources.get_distribution('WPEMPhase').location
    loc = os.path.join(extract_to_folder,'WPEMPhase')
    # supply CIF files 
    if os.path.isdir(os.path.join(loc,'strucs')) and os.path.isdir(os.path.join(loc,'pretrained')): pass
    else: 
        print("This is the first time CPICANN is being executed on your computer, configuring...")
        file_url = "https://figshare.com/ndownloader/files/45638907"
        _dir = os.path.join(loc, 'SystemFiles.zip')
        wget.download(file_url, _dir, bar=bar_progress)
        zipfile.ZipFile(_dir).extractall(loc)
        prefix_dir = os.path.join(loc, 'SystemFiles')
        zipfile.ZipFile(os.path.join(prefix_dir, 'strucs.zip')).extractall(loc)
        zipfile.ZipFile(os.path.join(prefix_dir, 'pretrained.zip')).extractall(loc)

    
    os.makedirs('figs', exist_ok=True)
    now = datetime.datetime.now()
    formatted_date_time = now.strftime('%Y-%m-%d %H:%M:%S')
    print(text2art("CPICANN"))
    print('The phase identification module of WPEM')
    print('URL : https://github.com/WPEM/CPICANN')
    print('Executed on :',formatted_date_time, ' | Have a great day.')  
    print('='*80)

    # get annotation data for display purposeys = np.zeros(4500)
    annoMap, elemMap = get_anno_map(loc)

    # get specific element setting map
    elem_setting_map = get_elem_setting()

    # choose the model to be loaded
    if Task == 'single-phase':
        load_path = os.path.join(loc,'pretrained','DPID_single-phase.pth') 
    elif Task == 'di-phase':
        load_path = os.path.join(loc,'pretrained','DPID_di-phase.pth')  
    else:
        raise ValueError('invalid inf_mode: {}'.format(Task))

    # load model parameters
    model = CPICANN_main(embed_dim=128, num_classes=23073)
    if Device == 'cpu':
        loaded = torch.load(load_path,map_location=torch.device('cpu'))
    else: loaded = torch.load(load_path)
    model.load_state_dict(loaded['model'])
    print('loaded model from {}'.format(load_path))
    if Device != 'cpu':
        model.to(Device)
    model.eval()

    # write the inference results to a csv file
    resFileName = 'infResults_{}.csv'.format(FilePath.split('/')[-1])
    dataFile = open(resFileName, 'w')
    dataWriter = csv.writer(dataFile)
    dataWriter.writerow(['path', 'fileName', 'predRank', 'pred', 'codId', 'formula', 'spaceGroupNo', 'spaceGroup'])

    fileList = os.listdir(FilePath)
    fileList.sort()
    for l in fileList:
        filePath = os.path.join(FilePath, l)
        print('\n>>>>>> RUNNING: {}'.format(filePath))

        data = convert_file(filePath)
        if data is None:
            continue

        logits = inference(model, data, Device).squeeze()

        try:
            elem_setting = elem_setting_map[l]

            # use specific element setting
            logits = filter_elem(logits, elemMap, elem_setting)
        except KeyError:
            # use global element setting
            # filter by include elements
            logits = filter_elem(logits, elemMap,
                                 {'include_must': ElementsSystem,
                                  'include_atLeastOne': ElementsContained,
                                  'exclude': ElementsExclude})

        predList = []
        if Task == 'single-phase':
            pred = logits.argmax().item()
            info = annoMap[pred]
            predList.append(info)
            conf = torch.nn.functional.softmax(logits, dim=0).max().item()

            print('pred cls_id : {}  confidence : {:.2f}%'.format(pred, conf * 100))
            print('pred cod_id : {}  formula : {}'.format(int(info[0]), info[2]))
            print('pred space group No: {}    space group : {}'.format(info[5], info[4]))
            dataWriter.writerow([FilePath, l, 1, pred, info[0], info[2], info[5], info[4]])
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
                dataWriter.writerow([FilePath, l, i+1, pred.item(), info[0], info[2], info[5], info[4]])

        _run(l, data, predList,loc)

    dataFile.close()

    print('\ninference result saved in {}'.format(resFileName))
    print('inference figures saved at figs/')
    print('THE END')
    return True



def inference(model, v, device):
    v = v / v.max() * 100

    v = torch.tensor(v, dtype=torch.float32).reshape(1, 1, -1)
    if device != 'cpu':
        v = v.to(device)
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


def get_anno_map(loc):
    vs = pd.read_csv(os.path.join(loc,'config','strucs.csv')).values
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

def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()