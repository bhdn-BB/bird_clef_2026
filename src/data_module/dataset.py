import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import hashlib


class AudioDataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            filepath_col: str,
            target_col: str,
            label2id: dict,
            feature_extractor,
            cache_dir: str = None,
            duration: float = 3.0,
            sample_rate: int = 32000,
            spectrogram_transform=None,
            is_test: bool = False,
            cache_mel: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.filepath_col = filepath_col
        self.target_col = target_col
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.feature_extractor = feature_extractor
        self.duration = duration
        self.sample_rate = sample_rate
        self.spectrogram_transform = spectrogram_transform
        self.is_test = is_test

        self.cache_dir = cache_dir
        self.cache_mel = cache_mel

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, filepath: str):
        h = hashlib.md5(filepath.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.pt")

    def _load_or_create_mel(self, filepath: str):

        if self.cache_dir is not None:
            path = self._cache_path(filepath)

            if os.path.exists(path):
                return torch.load(path)

        wave = self.feature_extractor.load_wave(
            filepath,
            duration=self.duration,
        ).squeeze(0)

        mel = self.feature_extractor.extract_mel_from_wave(
            wave.unsqueeze(0)
        )

        if self.cache_dir is not None and self.cache_mel:
            torch.save(mel, self._cache_path(filepath))

        return mel

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filepath = row[self.filepath_col]

        mel = self._load_or_create_mel(filepath)

        if self.spectrogram_transform is not None:
            mel = self.spectrogram_transform(mel)

        if self.is_test:
            return mel

        target = torch.zeros(self.num_classes, dtype=torch.bfloat16)

        labels = str(row[self.target_col]).split(";")
        for label in labels:
            label = label.strip()
            if label in self.label2id:
                target[self.label2id[label]] = 1.0

        return mel, target