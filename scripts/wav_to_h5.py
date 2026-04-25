import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

AUDIO_EXTENSIONS = {".ogg", ".wav", ".flac", ".mp3"}


def convert_file(src: str, dst: str, target_sr: int, normalize: bool) -> bool:
    dst_path = Path(dst)
    if dst_path.exists():
        return False
    try:
        wav, sr = torchaudio.load(src)
        wav = wav.mean(dim=0)
        if sr != target_sr:
            wav = T.Resample(sr, target_sr)(wav.unsqueeze(0)).squeeze(0)
        audio = wav.numpy().astype(np.float32)
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst_path, "w") as f:
            f.create_dataset("waveform", data=audio)
            f.attrs["sr"] = target_sr
    except Exception as e:
        print(f"\n[ERROR] {src}: {e}")
        return False
    return True


def discover_pairs(src_root: Path, dst_root: Path) -> list[tuple[str, str]]:
    return [
        (str(p), str((dst_root / p.relative_to(src_root)).with_suffix(".h5")))
        for p in src_root.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def run_parallel(pairs: list, workers: int, target_sr: int, normalize: bool) -> int:
    saved = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(convert_file, src, dst, target_sr, normalize): dst
            for src, dst in pairs
        }
        with tqdm(total=len(futures), desc="Converting") as bar:
            for fut in as_completed(futures):
                if fut.result():
                    saved += 1
                bar.update(1)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Convert audio files to HDF5 waveform format.")
    parser.add_argument("src_dir", type=str, help="Root directory with audio files (e.g. train_audio/)")
    parser.add_argument("dst_dir", type=str, help="Root directory to write .h5 files")
    parser.add_argument("--sr", type=int, default=32_000, help="Target sample rate (default: 32000)")
    parser.add_argument("--normalize", action="store_true", help="Peak-normalize each waveform before saving")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    pairs = discover_pairs(Path(args.src_dir), Path(args.dst_dir))
    print(f"Found {len(pairs)} audio files")

    saved = run_parallel(pairs, args.workers, args.sr, args.normalize)
    skipped = len(pairs) - saved
    print(f"Done. Converted {saved} files, skipped {skipped} (already existed).")


if __name__ == "__main__":
    main()
