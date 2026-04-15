import os
import torch
from tqdm import tqdm


def build_mel_cache(
    df,
    filepath_col: str,
    cache_dir: str,
    feature_extractor,
    duration: float,
):
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[Cache] Building cache in: {cache_dir}")

    unique_files = df[filepath_col].unique()

    for filepath in tqdm(unique_files):
        filename = os.path.basename(filepath)
        cache_path = os.path.join(cache_dir, f"{filename}.pt")

        if os.path.exists(cache_path):
            continue

        try:
            wave = feature_extractor.load_wave(
                filepath,
                duration=duration,
            ).squeeze(0)

            mel = feature_extractor.extract_mel_from_wave(
                wave.unsqueeze(0)
            )

            torch.save(mel, cache_path)

        except Exception as e:
            print(f"[Cache ERROR] {filepath}: {e}")