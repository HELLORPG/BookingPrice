# 从torch中的Dataset继承而来的

import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from data import InfoFile


class DFDataset(Dataset):
    def __init__(self, path_or_subset, use_subset=False):
        """
        初始化一个DataFrame形式的Dataset。
        :param path_or_subset: Data file path or a Subset instance.
        """
        if use_subset:
            # indices_len = len(path_or_subset.indices)
            sub_len = len(path_or_subset)
            # self.data = pd.concat([path_or_subset.dataset[path_or_subset.indices[i]] for i in range(0, indices_len)], ignore_index=True)
            self.data = pd.concat([path_or_subset[i] for i in range(0, sub_len)], ignore_index=True)
            # self.data = self.data.drop(self.data.columns[[0]], axis=1)
            # self.data = path_or_subset[:len(path_or_subset.indices)]
            # indices = path_or_subset.indices
            # self.data = path_or_subset.dataset[indices[0]]
            # for i in range(1, len(indices)):
            #     self.data = self.data.append(path_or_subset.dataset[indices[i]])
        else:
            self.data = InfoFile(path=path_or_subset).csv_to_dataframe()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item: item+1]

    def to_csv(self, path: str):
        """
        将数据内容存储到指定csv文件中。
        :param path: file path.
        :return: None
        """
        filedir, _ = os.path.split(path)
        os.makedirs(filedir, exist_ok=True)
        self.data.to_csv(path, sep=",", header=True, index=None)
        return


def split_train_val(input_path: str, train_path: str, val_path: str, train_ratio=0.0, train_size=0, val_size=0):
    """
    划分训练集和测试集。
    :param input_path: raw data path.
    :param train_path: train set data path.
    :param val_path: val set data path.
    :param train_ratio: train set ratio, from 0 to 1.
    :param train_size: train set size.
    :param val_size: val set size.
    :return: None
    """
    whole_dataset = DFDataset(input_path)

    assert train_ratio != 0.0 or train_size != 0 or val_size != 0
    whole_size = len(whole_dataset)
    if train_ratio != 0.0:
        train_size = int(whole_size * train_ratio)
        val_size = whole_size - train_size
    elif train_size != 0:
        val_size = whole_size - train_size
    else:
        train_size = whole_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(whole_dataset, [train_size, val_size])

    train_dir, _ = os.path.split(train_path)
    val_dir, _ = os.path.split(val_path)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_dataset = DFDataset(train_dataset, use_subset=True)
    val_dataset = DFDataset(val_dataset, use_subset=True)
    train_dataset.to_csv(train_path)
    val_dataset.to_csv(val_path)

    return


if __name__ == '__main__':
    split_train_val("../dataset/train.csv", "./split/train.csv", "./split/val.csv", val_size=5000)
