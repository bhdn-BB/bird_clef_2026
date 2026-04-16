import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        filepath_col: str,
        target_col: str,
        label2id: dict,
        cache_dir: str,
        spectrogram_transform=None,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.4,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.filepath_col = filepath_col
        self.target_col = target_col
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.cache_dir = cache_dir
        self.transform = spectrogram_transform

        self.mixup_p = mixup_p
        self.mixup_alpha = mixup_alpha
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _get_sample(self, idx: int):
        row = self.df.iloc[idx]
        filepath = row[self.filepath_col]
        start_sec = float(row.get("start", 0.0))
        cache_filename = f"{os.path.basename(filepath)}_{start_sec:.2f}.pt"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        try:
            mel = torch.load(cache_path, map_location="cpu", weights_only=True).float()
        except FileNotFoundError:
            raise RuntimeError(f"Cache missing: {cache_path}")

        if self.transform:
            mel = self.transform(mel)

        target = torch.zeros(self.num_classes, dtype=torch.float32)

        labels = str(row[self.target_col]).split(";")
        for label in labels:
            label = label.strip()
            if label in self.label2id:
                target[self.label2id[label]] = 1.0

        return mel, target

    def _apply_mixup(self, mel: torch.Tensor, target: torch.Tensor):
        if np.random.rand() >= self.mixup_p:
            return mel, target
        mix_idx = np.random.randint(0, len(self.df))
        mix_mel, mix_target = self._get_sample(mix_idx)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_mel = lam * mel + (1.0 - lam) * mix_mel
        mixed_target = lam * target + (1.0 - lam) * mix_target
        return mixed_mel, mixed_target

    def __getitem__(self, idx: int):
        mel, target = self._get_sample(idx)
        if self.is_train and self.mixup_p > 0:
            mel, target = self._apply_mixup(mel, target)
        return mel, target

