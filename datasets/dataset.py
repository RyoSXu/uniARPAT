import os
from torch.utils.data import Dataset
import numpy as np
import torch


class Dos_Dataset(Dataset):
    def __init__(self, data_dir="./data", split='train', dos_minmax = False, dos_zscore=False, scale_factor=1.0, apply_log=False, smear=0, choice=[],**kwargs) -> None:
        super().__init__()
        self.split = split
        self.smear = smear
        self.data_dir = data_dir+"/"+split+"/"
        
        self.elements  = self.get_elements()  #size (__len__, src_len)
        self.positions = self.get_positions() #size (__len__, src_len*3)
        data_len = self.positions.shape[0]

        if self.split == 'test_cif':
            self.edos_tgtdos = torch.zeros((data_len, 128), dtype=torch.float32)
            self.phdos_tgtdos = torch.zeros((data_len, 64), dtype=torch.float32)
        else:
            self.edos_tgtdos = self.get_dos_data(prefix="edos_tgtdos")
            self.phdos_tgtdos = self.get_dos_data(prefix="phdos_tgtdos")
        
        self.edos_mean = torch.mean(self.edos_tgtdos, dim=1, keepdim=True).float()
        self.edos_std = torch.std(self.edos_tgtdos, dim=1, keepdim=True).float()
        self.edos_min = torch.min(self.edos_tgtdos, dim=1, keepdim=True).values.float()
        self.edos_max = torch.max(self.edos_tgtdos, dim=1, keepdim=True).values.float()

        self.phdos_mean = torch.mean(self.phdos_tgtdos, dim=1, keepdim=True).float()
        self.phdos_std = torch.std(self.phdos_tgtdos, dim=1, keepdim=True).float()
        self.phdos_min = torch.min(self.phdos_tgtdos, dim=1, keepdim=True).values.float()
        self.phdos_max = torch.max(self.phdos_tgtdos, dim=1, keepdim=True).values.float()

        if scale_factor != 1.0:
            self.edos_tgtdos = self.edos_tgtdos * scale_factor
            self.phdos_tgtdos = self.phdos_tgtdos * scale_factor

        if apply_log:
            self.edos_tgtdos = torch.log1p(self.edos_tgtdos)
            self.phdos_tgtdos = torch.log1p(self.phdos_tgtdos)

        if dos_zscore:
            self.edos_tgtdos = (self.edos_tgtdos - self.edos_mean) / (self.edos_std + 1e-8)
            self.phdos_tgtdos = (self.phdos_tgtdos - self.phdos_mean) / (self.phdos_std + 1e-8)
        
        if dos_minmax:
            self.edos_tgtdos = (self.edos_tgtdos - self.edos_min) / (self.edos_max - self.edos_min + 1e-8)
            self.phdos_tgtdos = (self.phdos_tgtdos - self.phdos_min) / (self.phdos_max - self.phdos_min + 1e-8)

        if len(choice) != 0:
            cholist = torch.Tensor(choice).long()
            self.elements = self.elements.index_select(dim=0, index=cholist)
            self.positions = self.positions.index_select(dim=0, index=cholist)
            self.edos_tgtdos = self.edos_tgtdos.index_select(dim=0, index=cholist)
            self.phdos_tgtdos = self.phdos_tgtdos.index_select(dim=0, index=cholist)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        index = min(index, self.__len__() - 1)
        # 返回 10 个元素，包含所有归一化所需的参数
        return [
            self.elements[index],           # [0]
            self.positions[index].reshape(-1, 3), # [1]
            self.edos_tgtdos[index],        # [2]
            self.phdos_tgtdos[index],       # [3]
            self.edos_mean[index],          # [4]
            self.edos_std[index],           # [5]
            self.edos_min[index],           # [6]
            self.edos_max[index],           # [7]
            self.phdos_mean[index],         # [8]
            self.phdos_std[index],          # [9]
            self.phdos_min[index],          # [10]
            self.phdos_max[index]           # [11]
        ]

    def get_elements(self):
        filename = os.path.join(self.data_dir, f"elements_{self.split}.npy")
        return torch.from_numpy(np.load(filename)).long()

    def get_positions(self):
        filename = os.path.join(self.data_dir, f"positions_{self.split}.npy")
        return torch.from_numpy(np.load(filename)).float()

    def get_dos_data(self, prefix):
        filename = os.path.join(self.data_dir, f"{prefix}_{self.split}.npy")
        return torch.from_numpy(np.load(filename)).float()

if __name__ == "__main__":
    test = Dos_Dataset(data_dir="./data/train4ARPAT", split="train")
    print(test.__getitem__(15))
