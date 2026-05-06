import argparse
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torchaudio
import torchaudio.transforms as T

from joblib import Parallel, delayed
from tqdm import tqdm


AUDIO_EXTENSIONS = {".ogg", ".wav", ".flac", ".mp3"}


def convert_single_file(
    src: str,
    dst: str,
    target_sr: int,
    normalize: bool = False,
) -> int:
    """
    Returns:
        1  -> saved
        0  -> skipped
        -1 -> error
    """
    dst_path = Path(dst)

    if dst_path.exists():
        return 0

    try:
        wav, sr = torchaudio.load(src)

        # mono
        wav = wav.mean(dim=0)

        # resample
        if sr != target_sr:
            wav = T.Resample(sr, target_sr)(wav.unsqueeze(0)).squeeze(0)

        audio = wav.numpy().astype(np.float32)

        # normalize
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(dst_path, "w") as f:
            f.create_dataset("waveform", data=audio)
            f.attrs["sr"] = target_sr

        return 1

    except Exception as e:
        print(f"\n[ERROR] {src}: {e}")
        return -1


def discover_pairs(src_root: Path, dst_root: Path) -> List[Tuple[str, str]]:
    return [
        (
            str(p),
            str((dst_root / p.relative_to(src_root)).with_suffix(".h5")),
        )
        for p in src_root.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def run_parallel(
    pairs: List[Tuple[str, str]],
    workers: int,
    target_sr: int,
    normalize: bool,
) -> None:
    print(f"[Joblib] Running with {workers} workers...")

    results: List[int] = Parallel(n_jobs=workers, backend="loky")(
        delayed(convert_single_file)(src, dst, target_sr, normalize)
        for src, dst in tqdm(pairs, desc="Queueing tasks")
    )

    saved = sum(1 for r in results if r == 1)
    skipped = sum(1 for r in results if r == 0)
    errors = sum(1 for r in results if r == -1)

    print(
        f"\n[Done] "
        f"Saved: {saved}, "
        f"Skipped: {skipped}, "
        f"Errors: {errors}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to HDF5 waveform format."
    )
    parser.add_argument("src_dir", type=str)
    parser.add_argument("dst_dir", type=str)
    parser.add_argument("--sr", type=int, default=32_000)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())

    args = parser.parse_args()

    pairs = discover_pairs(Path(args.src_dir), Path(args.dst_dir))
    print(f"Found {len(pairs)} audio files")

    run_parallel(pairs, args.workers, args.sr, args.normalize)


if __name__ == "__main__":
    main()