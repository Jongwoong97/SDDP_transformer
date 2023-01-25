import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SddpDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = [torch.FloatTensor(x) for x in x_data]
        self.x_data = pad_sequence(self.x_data).transpose(0, 1)

        self.y_data = [torch.FloatTensor(y) for y in y_data]
        self.y_data = pad_sequence(self.y_data).transpose(0, 1)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def max_seq_length(self):
        return self.y_data.shape[1]