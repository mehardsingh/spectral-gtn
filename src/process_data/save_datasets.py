from eeg_dataset import EEGDataset
import argparse
import torch
from torch.utils.data import random_split
import os

def save_datasets(raw_dir, save_dir, motion_type):
    eeg_ds = EEGDataset(raw_dir, motion_type)
    
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(eeg_ds, [0.9, 0.1], generator=generator)
    train_ds, val_ds = random_split(train_ds, [0.88, 0.12], generator=generator)

    print(f"Train DS: {int(100*len(train_ds)/len(eeg_ds))}%")
    print(f"Val DS: {int(100*len(val_ds)/len(eeg_ds))}%")
    print(f"Test DS: {int(100*len(test_ds)/len(eeg_ds))}%")

    torch.save(train_ds, os.path.join(save_dir, 'train_ds.pth'))
    torch.save(val_ds, os.path.join(save_dir, 'val_ds.pth'))
    torch.save(test_ds, os.path.join(save_dir, 'test_ds.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Saving train/val/test splits of EEG data')
    parser.add_argument('--raw_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--motion_type')
    args = parser.parse_args()

    save_datasets(args.raw_dir, args.save_dir, args.motion_type)