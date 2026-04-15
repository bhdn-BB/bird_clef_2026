import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class AudioDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            filepath_col: str,
            target_col: str,
            label2id: dict,
            cache_dir: str,
            spectrogram_transform=None,
            is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.filepath_col = filepath_col
        self.target_col = target_col
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.cache_dir = cache_dir
        self.spectrogram_transform = spectrogram_transform
        self.is_test = is_test


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row[self.filepath_col]

        cache_path = os.path.join(
            self.cache_dir,
            f'{os.path.basename(filepath)}.pt'
        )

        if not os.path.exists(cache_path):
            raise RuntimeError(f"Cache missing: {cache_path}")

        mel = torch.load(cache_path)

        if self.spectrogram_transform is not None:
            mel = self.spectrogram_transform(mel)

        if self.is_test:
            return mel

        target = torch.zeros(self.num_classes, dtype=torch.float32)

        labels = str(row[self.target_col]).split(";")
        for label in labels:
            label = label.strip()
            if label in self.label2id:
                target[self.label2id[label]] = 1.0

        return mel, target