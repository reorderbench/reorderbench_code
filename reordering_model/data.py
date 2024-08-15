import os

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate, get_worker_info
from torchvision import transforms


def main_key(key: str):
    return key[:key.rindex('_')]

def no_swap_key(key: str):
    key_list = key.split('_')
    key_list[key_list.index('swap') + 1] = '0'
    return '_'.join(key_list)

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset

    if isinstance(dataset, BaseDataset):
        if dataset.data_dir.endswith('.npz'):
            if os.path.exists(dataset.data_dir):
                dataset.dataset = np.load(dataset.data_dir, allow_pickle=True)
            else:
                data_dir = dataset.data_dir[:-4]
                i = 0
                dataset_names = []
                while os.path.exists(f"{data_dir}_{i}.npz"):
                    dataset_names.append(f"{data_dir}_{i}.npz")
                    i += 1
                dataset.dataset = []
                for i, dataset_name in enumerate(dataset_names):
                    dataset.dataset.append(np.load(dataset_name, allow_pickle=True))
        else:
            raise NotImplementedError
            # print(f"Reopened npz file for worker {worker_id}")

def test_collate(batch):
    input_matrix = default_collate([item[0] for item in batch])
    target_matrix = default_collate([item[1] for item in batch])
    # score_upb and mat_pattern cannot be default collated
    score_upb = [item[2] for item in batch]
    mat_pattern = [item[3] for item in batch]
    return input_matrix, target_matrix, score_upb, mat_pattern

class BaseDataset(Dataset):
    def __init__(self, data_dir: str, indices=None):
        self.data_dir = data_dir

        if data_dir.endswith('.npy'):
            raise NotImplementedError
            self.dataset = np.load(data_dir, allow_pickle=True).item()
            keys = list(self.dataset.keys())
        elif data_dir.endswith('.npz'):
            if os.path.exists(data_dir):
                self.dataset = np.load(data_dir, allow_pickle=True)
                keys = list(self.dataset.files)
                print(f"Loaded {data_dir}")
            else:
                print(f"Data file {data_dir} does not exist. Trying to load multiple files.")
                dataset_names = []
                data_dir = data_dir[:-4]
                i = 0
                while os.path.exists(f"{data_dir}_{i}.npz"):
                    dataset_names.append(f"{data_dir}_{i}.npz")
                    i += 1
                keys = []
                self.dataset = []
                for i, dataset_name in enumerate(dataset_names):
                    self.dataset.append(np.load(dataset_name, allow_pickle=True))
                    keys += list(self.dataset[i].files)
                    print(f"Loaded {dataset_name}")

        else:
            raise NotImplementedError

        if indices is not None:
            keys = [keys[i] for i in indices]

        # Keys without score
        self.keys = [main_key(k) for k in keys]
        # Key without score -> original key
        self.key_map = {main_key(k): k for k in keys}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = self.from_key(key)
        return self.process(data, key)

    def from_key(self, key):
        key = self.key_map[key]
        if isinstance(self.dataset, list):
            for i in range(len(self.dataset)):
                if key in self.dataset[i].files:
                    return self.dataset[i][key].item()
        elif isinstance(self.dataset, dict):
            return self.dataset[key]
        else:
            return self.dataset[key].item()

    def process(self, data, key):
        raise NotImplementedError

    def get_score_upb(self, key):
        try:
            return self.from_key(no_swap_key(key))['score']
        except KeyError:
            return None

class MatrixDataset(BaseDataset):
    def __init__(self, data_dir, indices=None):
        super(MatrixDataset, self).__init__(data_dir, indices)

    def process(self, data, key):
        return (
            transforms.ToTensor()(data['swapped_noise_mat']).to(torch.float), # input_matrix
            transforms.ToTensor()(data['noise_mat']).to(torch.float).squeeze(), # target_matrix
        )

class TestMatrixDataset(BaseDataset):
    def __init__(self, data_dir, indices=None):
        super(TestMatrixDataset, self).__init__(data_dir, indices)

    def process(self, data, key):
        return (
            transforms.ToTensor()(data['swapped_noise_mat']).to(torch.float), # input_matrix
            transforms.ToTensor()(data['noise_mat']).to(torch.float).squeeze(), # target_matrix
            self.get_score_upb(key), # score_upb
            data['mat_pattern'], # mat_pattern
        )

class TestContinuousMatrixDataset(BaseDataset):
    def __init__(self, data_dir, indices = None):
        super(TestContinuousMatrixDataset, self).__init__(data_dir, indices)

    def process(self, data, key):
        return (
            transforms.ToTensor()(data['swapped_noise_mat']).to(torch.float), # input_matrix
            transforms.ToTensor()(data['noise_mat']).to(torch.float).squeeze(), # target_matrix
            self.get_score_upb(key), # score_upb
            data['mat_pattern'], # mat_pattern
            # data.get('period', None), # periods (is this necessary?)
        )

