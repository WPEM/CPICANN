import numpy as np
import pandas as pd
from scipy import interpolate

global dataWriter

def convert_file(file_path):
    suffix = file_path.split('.')[-1]
    if suffix not in ['txt', 'csv', 'xy']:
        Warning(f'File {file_path} not supported, skiping...')
        return None

    if suffix == 'txt':
        return txt_to_csv(file_path)
    elif suffix == 'csv':
        return csv_to_csv(file_path)
    elif suffix == 'xy':
        return xy_to_csv(file_path)

def txt_to_csv(file_path):
    f = open(file_path, 'r')
    rows = []
    for line in f.readlines():
        line = line.strip('\n')
        line = line.replace('\t', ' ')
        line = [x for x in line.split(' ') if x != '']
        if len(line) == 3:
            try:
                line = [line[0], float(line[1])-float(line[2])]
            except ValueError:
                continue
        elif len(line) < 2 or len(line) > 3:
            continue
        rows.append(line)
    f.close()

    outData = upsample(rows)
    return outData

def csv_to_csv(file_path):
    fromData = pd.read_csv(file_path).values
    outData = upsample(list(fromData))
    return outData

def xy_to_csv(file_path):
    return txt_to_csv(file_path)

def upsample(rows):
    if len(rows) == 0:
        Warning('Empty data!')
        return None

    rows.insert(0, ['10', rows[0][1]]) if float(rows[0][0]) > 10 else None
    rows.append(['80', rows[-1][1]]) if float(rows[-1][0]) < 80 else None
    rowsData = np.array(rows, dtype=np.float32)
    f = interpolate.interp1d(rowsData[:, 0], rowsData[:, 1], kind='slinear')
    xnew = np.linspace(10, 80, 4500)
    ynew = f(xnew)
    # outData = np.array([xnew, ynew]).T
    return ynew
