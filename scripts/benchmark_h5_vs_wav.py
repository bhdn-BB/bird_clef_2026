"""
Benchmark: h5 waveform cache vs raw OGG read speed.

Both paths end at the same point (mel spectrogram tensor) so the
comparison is apples-to-apples with what the DataLoader actually does.

Usage (Kaggle notebook cell):
    !git clone https://github.com/YOUR/bird_clef_2026 && pip install h5py torchaudio -q
    %run bird_clef_2026/scripts/benchmark_h5_vs_wav.py
"""

import os
import time
import random
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

AUDIO_ROOT = "/kaggle/input/birdclef-2026/train_audio"
H5_ROOT = "/kaggle/working/h5_benchmark"
MEL_CACHE_ROOT = "/kaggle/working/mel_benchmark"
SR = 32_000
DURATION = 3.0
N_SAMPLES = 200
SEED = 42

random.seed(SEED)

mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=8000
)
to_db = T.AmplitudeToDB(stype="power", top_db=80)
target_frames = int(DURATION * SR)


# matches dataset._load_from_cache: torch.load pre-computed .pt mel
def read_mel_cache(pt_path: str) -> torch.Tensor:
    return torch.load(pt_path, map_location="cpu", weights_only=True).float()


# matches dataset._load_from_h5 (without wave_transform): h5 read + mel compute
def read_h5_to_mel(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        sr = int(f.attrs["sr"])
        end = int(DURATION * sr)
        wave = torch.from_numpy(f["waveform"][:end].astype(np.float32))
    if wave.shape[0] < target_frames:
        wave = torch.nn.functional.pad(wave, (0, target_frames - wave.shape[0]))
    return to_db(mel_transform(wave.unsqueeze(0)))


# raw OGG baseline: decode + resample + mel compute (no cache at all)
def read_ogg_to_mel(ogg_path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(ogg_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = T.Resample(sr, SR)(wav)
    wav = wav[:, :target_frames]
    if wav.shape[1] < target_frames:
        wav = torch.nn.functional.pad(wav, (0, target_frames - wav.shape[1]))
    return to_db(mel_transform(wav))


def make_h5(src: str, dst: str) -> None:
    if Path(dst).exists():
        return
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    wav, sr = torchaudio.load(src)
    wav = wav.mean(dim=0)
    if sr != SR:
        wav = T.Resample(sr, SR)(wav.unsqueeze(0)).squeeze(0)
    with h5py.File(dst, "w") as f:
        f.create_dataset("waveform", data=wav.numpy().astype(np.float32))
        f.attrs["sr"] = SR


def make_mel_cache(src: str, dst: str) -> None:
    if Path(dst).exists():
        return
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    mel = read_ogg_to_mel(src)
    torch.save(mel, dst)


def bench(label: str, fn, paths: list) -> float:
    print(f"Benchmarking {label} ({len(paths)} samples)...")
    t0 = time.perf_counter()
    for p in paths:
        fn(p)
    elapsed = time.perf_counter() - t0
    return elapsed


# ── 1. pick sample files ───────────────────────────────────────────────────────
all_oggs = glob(os.path.join(AUDIO_ROOT, "*", "*.ogg"))
assert len(all_oggs) > 0, f"No .ogg files found under {AUDIO_ROOT}"
sample_oggs = random.sample(all_oggs, min(N_SAMPLES, len(all_oggs)))

sample_h5s = [
    str((Path(H5_ROOT) / Path(p).relative_to(AUDIO_ROOT)).with_suffix(".h5"))
    for p in sample_oggs
]
sample_pts = [
    str((Path(MEL_CACHE_ROOT) / Path(p).relative_to(AUDIO_ROOT)).with_suffix(".pt"))
    for p in sample_oggs
]

# ── 2. build both caches for the sample ───────────────────────────────────────
print(f"Building h5 and mel caches for {len(sample_oggs)} files...")
t0 = time.perf_counter()
for src, h5, pt in zip(sample_oggs, sample_h5s, sample_pts):
    make_h5(src, h5)
    make_mel_cache(src, pt)
print(f"  done in {time.perf_counter() - t0:.1f}s\n")

# ── 3. benchmark all three paths ──────────────────────────────────────────────
mel_transform(torch.zeros(1, target_frames))  # one-time init before timing
mel_time = bench("mel cache (.pt)  [current, no wave aug]", read_mel_cache, sample_pts)
h5_time  = bench("h5 waveform      [new, wave aug possible]", read_h5_to_mel, sample_h5s)
ogg_time = bench("raw OGG          [no cache baseline]", read_ogg_to_mel, sample_oggs)

# ── 4. report ─────────────────────────────────────────────────────────────────
def row(label, t):
    return f"  {label:<38} {t:.2f}s  ({t/N_SAMPLES*1000:.1f} ms/sample)"

print("\n── Results ───────────────────────────────────────────────────────")
print(row("mel cache (.pt)  [current, no wave aug]", mel_time))
print(row("h5 waveform      [new, wave aug possible]", h5_time))
print(row("raw OGG          [no cache baseline]", ogg_time))
print(f"\n  h5 vs mel cache cost:  {h5_time/mel_time:.2f}x slower  (overhead of on-the-fly mel)")
print(f"  h5 vs raw OGG gain:    {ogg_time/h5_time:.2f}x faster  (benefit of pre-decoding)")
print("──────────────────────────────────────────────────────────────────")
