import os
from typing import Dict

import librosa
import pandas as pd

from tqdm import tqdm


tqdm.pandas()


def get_label2id(taxonomy_df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    return {
        label: idx
        for idx, label in enumerate(taxonomy_df[label_col].tolist())
    }

def merge_dataframes(
        cleaned_df: pd.DataFrame,
        soundscapes_df: pd.DataFrame,
        cleaned_audio_dir: str,
        soundscapes_audio_dir: str,
) -> pd.DataFrame:

    cleaned_df['filepath'] = cleaned_df['filename'].apply(
        lambda path: os.path.join(cleaned_audio_dir, path)
    )
    cleaned_df.drop(columns=['filename'], inplace=True)
    cleaned_df['start'] = 0.0
    cleaned_df['end'] = cleaned_df['filepath'].progress_apply(
        lambda path: librosa.get_duration(path=path)
    )
    cleaned_df = cleaned_df[['primary_label', 'start', 'end', 'filepath']]

    soundscapes_df['filepath'] = soundscapes_df['filename'].apply(
        lambda path: os.path.join(soundscapes_audio_dir, path)
    )
    soundscapes_df.drop(columns=['filename'], inplace=True)
    soundscapes_df['start'] = pd.to_timedelta(soundscapes_df['start']).dt.total_seconds()
    soundscapes_df['end'] = pd.to_timedelta(soundscapes_df['end']).dt.total_seconds()
    combined_df = pd.concat([cleaned_df, soundscapes_df], ignore_index=True)
    return combined_df

def split_audio_samples(combined_df: pd.DataFrame, max_duration: float) -> pd.DataFrame:

    new_rows = []

    combined_df_loader = tqdm(
        combined_df.iterrows(),
        total=len(combined_df),
        desc="Splitting audio samples"
    )

    for _, row in combined_df_loader:

        start = row['start']
        end = row['end']

        while start < end:
            seg_end = min(start + max_duration, end)
            new_row = row.copy()
            new_row['start'] = start
            new_row['end'] = seg_end
            new_rows.append(new_row)
            start = seg_end

    split_df = pd.DataFrame(new_rows).reset_index(drop=True)
    return split_df

