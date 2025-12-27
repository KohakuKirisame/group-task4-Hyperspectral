import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, offset=None):

        self.data = np.load(data_path, mmap_mode="r")
        if offset is not None:
            self.data = self.data[offset:]
        if label_path is not None:
            self.label = np.load(label_path, mmap_mode="r")
            if offset is not None:
                self.label = self.label[offset:]
        # if not torch.is_tensor(features):
        #     features = torch.tensor(features, dtype=torch.float32)



    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.label[idx], dtype=torch.long)


def  build_dataloader(data_path,
                     label_path,
                     batch_size=25565,
                     shuffle=True,
                     num_workers=4,
                     drop_last=True,
                     pin_memory=True,
                     offset=None):
    dataset = CustomDataset(data_path, label_path, offset)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=drop_last,
                      pin_memory=pin_memory)


class EvalDataset(Dataset):
    def __init__(self, feature_path):
        self.features = np.load(feature_path, mmap_mode="r")
        # self.ids = np.load(id_path, mmap_mode="r")

        # assert len(self.features) == len(self.ids)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        sample_id = idx
        return sample_id, x
