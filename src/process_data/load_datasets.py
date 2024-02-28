import torch
import os

def load_datasets(load_dir):
    train_ds = torch.load(os.path.join(load_dir, 'train_ds.pth'))
    val_ds = torch.load(os.path.join(load_dir, 'val_ds.pth'))
    test_ds = torch.load(os.path.join(load_dir, 'test_ds.pth'))

    return train_ds, val_ds, test_ds
