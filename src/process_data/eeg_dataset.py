import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset

def load_file(filepath):
    df = pd.read_csv(filepath)

    m1 = np.array(df["M1"])
    sma = np.array(df["SMA"])
    lpfc = np.array(df["lPFC"])
    rpfc = np.array(df["rPFC"])

    all_data = np.array([m1, sma, lpfc, rpfc]).T
    epoch_data = np.reshape(all_data, (-1, 501, 4))

    groups = np.array(df["GroupID"])
    groups = np.reshape(groups, (-1, 501))

    labels = np.array(df["ML Label"])
    mapping_dict = {"ROM": 0, "Speed": 1}
    labels = np.vectorize(mapping_dict.get)(labels)
    labels = np.reshape(labels, (-1, 501))

    return epoch_data, groups, labels

def load_all(data_dir, motion_type):
    epoch_data = list()
    groups = list()
    labels = list()

    if motion_type == "both":
        filepaths = os.listdir(data_dir)
    else:
        filepaths = os.listdir(data_dir)
        filepaths = [fname for fname in filepaths if motion_type in fname]

    for filepath in tqdm(filepaths, desc="Loading data"):
        f_epoch_data, f_groups, f_labels = load_file(os.path.join(data_dir, filepath))
        
        epoch_data.append(f_epoch_data)
        groups.append(f_groups)
        labels.append(f_labels)

    epoch_data = np.concatenate(epoch_data, axis=0)
    groups = np.concatenate(groups, axis=0)[:,0]
    labels = np.concatenate(labels, axis=0)[:,0]

    return epoch_data, groups, labels

class EEGDataset(Dataset):
    def __init__(self, data_dir, motion_type):
        self.epoch_data, self.groups, self.labels = load_all(data_dir, motion_type)
    
    def __len__(self):
        return self.epoch_data.shape[0]
    
    def __getitem__(self, index):
        epoch_data = self.epoch_data[index]
        group = self.groups[index]
        label = self.labels[index]
        
        return epoch_data, group, label