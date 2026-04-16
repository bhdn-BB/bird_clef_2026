import os
import pandas as pd
import torch
from tqdm import tqdm

from src.data_module.wave_features_extractor import WaveFeaturesExtractor


def build_mel_cache(
        df: pd.DataFrame,
        filepath_col: str,
        cache_dir: str,
        feature_extractor: WaveFeaturesExtractor,
        duration: float,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching Mel Spectrograms"):

        filepath = row[filepath_col]
        start_sec = float(row.get('start', 0.0))
        cache_filename = f"{os.path.basename(filepath)}_{start_sec:.4f}.pt"
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path):
            continue

        try:
            wave = feature_extractor.load(
                filepath,
                offset=start_sec,
                duration=duration,
            )
            mel = feature_extractor.to_mel(wave)
            torch.save(mel, cache_path)
        except Exception as e:
            print(f"[Cache ERROR] {filepath} at {start_sec}s: {e}")
