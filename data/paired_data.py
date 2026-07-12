import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def load_img(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0   # HWC
    img = torch.from_numpy(img).permute(2, 0, 1)    # CHW
    return img

class PairDataset(Dataset):
    def __init__(self, inp_path, gt_path=None, train=False, crop_size=256):
        super().__init__()
        self.inp_path = inp_path
        self.gt_path = gt_path
        self.train = train
        self.crop_size = crop_size
        self.img_li = sorted(os.listdir(inp_path))

    def _paired_aug(self, inp, gt):
        _, h, w = inp.shape
        ps = self.crop_size
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        inp = inp[:, top:top + ps, left:left + ps]
        gt = gt[:, top:top + ps, left:left + ps]
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[2])
            gt = torch.flip(gt, dims=[2])
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[1])
            gt = torch.flip(gt, dims=[1])
        k = random.randint(0, 3)
        if k:
            inp = torch.rot90(inp, k, dims=[1, 2])
            gt = torch.rot90(gt, k, dims=[1, 2])
        return inp, gt

    def __getitem__(self, index):
        name = self.img_li[index]
        inp = load_img(os.path.join(self.inp_path, name))
        if self.gt_path is not None:
            gt = load_img(os.path.join(self.gt_path, name))
            if self.train:
                inp, gt = self._paired_aug(inp, gt)
            return inp, gt, os.path.splitext(name)[0]
        return inp, os.path.splitext(name)[0]
    def __len__(self):
        return len(self.img_li)


def get_train_loader(vi_train_path, ir_train_path, batch_size=8, num_workers=4, crop_size=256):
    train_data = PairDataset(inp_path=vi_train_path,gt_path=ir_train_path,train=True,crop_size=crop_size)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader

def get_test_loader(vi_val_path, ir_val_path, batch_size=1, num_workers=0):
    test_data = PairDataset(inp_path=vi_val_path,gt_path=ir_val_path,train=False)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return test_loader