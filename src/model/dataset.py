import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class XrdDataset(Dataset):
    def __init__(self, data_dir, annotations_file):
        self.labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dataid = str(self.labels.iloc[idx, 0])
        data_path = os.path.join(self.data_dir, dataid + '.csv')
        data_csv = pd.read_csv(data_path)
        data = data_csv.values.astype(np.float32).T

        label = self.labels.iloc[idx, 1]

        return data, label


class mixDataset_cls_dynamic(Dataset):
    def __init__(self, data_dir, anno_struc, mode):
        self.data_dir = data_dir
        self.codIdList = pd.read_csv(anno_struc).values[:, 0].astype(np.int32)
        self.mode = mode

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        choice1, choice2 = np.random.randint(0, 23073, 2)
        if self.mode == 'train':
            rand1, rand2 = np.random.randint(1, 25, 2)
        else:
            rand1, rand2 = np.random.randint(1, 7, 2)
        data_path1 = os.path.join(self.data_dir, '{}_{}.csv'.format(self.codIdList[choice1], rand1))
        data_path2 = os.path.join(self.data_dir, '{}_{}.csv'.format(self.codIdList[choice2], rand2))
        data1 = pd.read_csv(data_path1).values.astype(np.float32).T
        data2 = pd.read_csv(data_path2).values.astype(np.float32).T

        ratio1 = np.random.randint(20, 81)
        ratio2 = 100 - ratio1

        label = np.zeros(23073).astype(np.float32)

        label[choice1] = 0.4
        label[choice2] = 0.4

        return data1, data2, ratio1, ratio2, label
