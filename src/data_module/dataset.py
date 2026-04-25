import os
from pathlib import Path

import h5py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        filepath_col: str,
        target_col: str,
        label2id: dict,
        # mel cache mode
        cache_dir: str = None,
        # h5 mode — provide all four or none
        h5_dir: str = None,
        audio_root: str = None,
        feature_extractor=None,
        duration: float = None,
        wave_transform=None,
        # shared
        spectrogram_transform=None,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.4,
        is_train: bool = True,
    ):
        if h5_dir is None and cache_dir is None:
            raise ValueError("Provide either cache_dir (mel cache) or h5_dir (waveform h5).")
        if h5_dir is not None and (audio_root is None or feature_extractor is None or duration is None):
            raise ValueError("h5_dir requires audio_root, feature_extractor, and duration.")

        self.df = df.reset_index(drop=True)
        self.filepath_col = filepath_col
        self.target_col = target_col
        self.label2id = label2id
        self.num_classes = len(label2id)

        self.cache_dir = cache_dir
        self.h5_dir = h5_dir
        self.audio_root = audio_root
        self.feature_extractor = feature_extractor
        self.duration = duration
        self.wave_transform = wave_transform
        self.spectrogram_transform = spectrogram_transform

        self.mixup_p = mixup_p
        self.mixup_alpha = mixup_alpha
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_from_h5(self, filepath: str, start_sec: float) -> torch.Tensor:
        rel = Path(filepath).relative_to(self.audio_root)
        h5_path = Path(self.h5_dir) / rel.with_suffix(".h5")
        with h5py.File(h5_path, "r") as f:
            sr = int(f.attrs["sr"])
            start = int(start_sec * sr)
            end = start + int(self.duration * sr)
            wave = f["waveform"][start:end].astype(np.float32)

        target_len = int(self.duration * sr)
        if len(wave) < target_len:
            wave = np.pad(wave, (0, target_len - len(wave)))

        if self.wave_transform is not None:
            wave = self.wave_transform(samples=wave, sample_rate=sr)

        return self.feature_extractor.to_mel(torch.from_numpy(wave).unsqueeze(0))

    def _load_from_cache(self, filepath: str, start_sec: float) -> torch.Tensor:
        cache_filename = f"{os.path.basename(filepath)}_{start_sec:.2f}.pt"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=True).float()
        except FileNotFoundError:
            raise RuntimeError(f"Cache missing: {cache_path}")

    def _get_sample(self, idx: int):
        row = self.df.iloc[idx]
        filepath = row[self.filepath_col]
        start_sec = float(row.get("start", 0.0))

        if self.h5_dir is not None:
            mel = self._load_from_h5(filepath, start_sec)
        else:
            mel = self._load_from_cache(filepath, start_sec)

        if self.spectrogram_transform:
            mel = self.spectrogram_transform(mel)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in str(row[self.target_col]).split(";"):
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
        return lam * mel + (1.0 - lam) * mix_mel, lam * target + (1.0 - lam) * mix_target

    def __getitem__(self, idx: int):
        mel, target = self._get_sample(idx)
        if self.is_train and self.mixup_p > 0:
            mel, target = self._apply_mixup(mel, target)
        return mel, target
