import os
import torch
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm


def process_single_sample(
    row_data,
    filepath_col,
    cache_dir,
    feature_extractor,
    duration,
):
    filepath = row_data[filepath_col]
    start_sec = float(row_data.get("start", 0.0))

    cache_filename = f"{os.path.basename(filepath)}_{start_sec:.2f}.pt"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 0 — skipped
    if os.path.exists(cache_path):
        return 0

    try:
        wave = feature_extractor.load(
            filepath,
            offset=start_sec,
            duration=duration,
        )
        mel = feature_extractor.to_mel(wave)

        torch.save(mel, cache_path)

        # 1 — saved
        return 1

    except Exception as e:
        print(f"\n[ERROR] {filepath} at {start_sec}s: {e}")

        # -1 — error
        return -1


def build_mel_cache(
    df: pd.DataFrame,
    filepath_col: str,
    cache_dir: str,
    feature_extractor,
    duration: float,
    n_workers: int = 8,
):
    os.makedirs(cache_dir, exist_ok=True)

    rows = df.to_dict("records")

    print(f"[Joblib] Starting parallel caching with {n_workers} workers...")

    results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(process_single_sample)(
            row,
            filepath_col,
            cache_dir,
            feature_extractor,
            duration,
        )
        for row in tqdm(rows, desc="Queueing tasks")
    )

    saved = sum(1 for r in results if r == 1)
    skipped = sum(1 for r in results if r == 0)
    errors = sum(1 for r in results if r == -1)

    print(
        f"\n[Cache Done] "
        f"Saved: {saved}, "
        f"Skipped: {skipped}, "
        f"Errors: {errors}"
    )
