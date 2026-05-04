# BirdCLEF 2026

Bird sound classification for the BirdCLEF 2026 competition.

## Training

```bash
python train.py --config <path/to/experiment.yml> [--accelerator gpu] [--devices 1] [--precision 32]
```

`--accelerator`, `--devices`, `--precision` are runtime/hardware options — they are never stored in YAML configs.

### Experiments

```bash
# No augmentation (offline mel cache)
python train.py --config config/experiment_no_aug/experiment.yml

# Spectrogram augmentation — SpecAugment (offline mel cache)
python train.py --config config/experiment_mel_aug/experiment.yml

# Waveform augmentation — noise / pitch / stretch (online mel from HDF5)
python train.py --config config/experiment_wave_aug/experiment.yml

# Wave + spectrogram augmentation (online mel from HDF5)
python train.py --config config/experiment_both_aug/experiment.yml

# Wave + spectrogram + Mixup (online mel from HDF5)
python train.py --config config/experiment_all_aug/experiment.yml
```

WandB credentials can be passed via CLI options or environment variables:
```bash
python train.py --config config/experiment_mel_aug/experiment.yml \
  --wandb-project birdclef2026 \
  --wandb-entity my-team \
  --wandb-api-key <key>
# or set WANDB_PROJECT / WANDB_ENTITY / WANDB_API_KEY in environment
```

---

## Config structure

```
config/
  global.yml                    # shared across all experiments
  experiment_no_aug/
    experiment.yml
  experiment_mel_aug/
    experiment.yml
    augs.yml
  experiment_wave_aug/
    experiment.yml
    augs.yml
  experiment_both_aug/
    experiment.yml
    augs.yml
  experiment_all_aug/
    experiment.yml
    augs.yml
```

The variant (no_aug / mel_aug / wave_aug / both_aug / all_aug) is detected automatically based on which augmentation keys are present in `augs.yml`:

| Keys in `augs.yml` | Variant |
|---|---|
| *(no file)* | `no_aug` |
| `mel_augmentation` only | `mel_aug` |
| `wave_augmentation` only | `wave_aug` |
| `mel_augmentation` + `wave_augmentation` | `both_aug` |
| `mel_augmentation` + `wave_augmentation` + `mixup` | `all_aug` |

---

## Config reference

### `config/global.yml`

Shared by all experiments. Do not copy into experiment directories.

| Key | Description |
|---|---|
| `data.cleaned_df_path` | Path to `train.csv` |
| `data.taxonomy_df_path` | Path to `taxonomy.csv` |
| `data.cleaned_audio_dir` | Directory with training audio |
| `data.filepath_col` | Column name for file paths in the CSV |
| `data.target_col` | Column name for labels |
| `data.cache_dir` | Mel-spectrogram cache dir (offline pipeline) |
| `data.h5_dir` | HDF5 waveform cache dir (online pipeline) |
| `data.audio_root` | Root prefix for audio paths in HDF5 pipeline |
| `data.sample_rate` | Target sample rate (Hz) |
| `data.duration` | Clip duration (seconds) |
| `mel_dim.sr` | Sample rate for mel transform |
| `mel_dim.n_mels` | Number of mel bins |
| `mel_dim.hop_length` | STFT hop length |
| `mel_dim.n_fft` | FFT size |
| `mel_dim.freq_min` | Minimum frequency (Hz) |
| `mel_dim.freq_max` | Maximum frequency (Hz) |
| `mel_dim.db_delta` | top_db for amplitude_to_db |
| `seed` | Global random seed |
| `val_split` | Fraction of data held out for validation |
| `num_workers` | DataLoader workers |

### `experiment.yml`

Experiment-specific hyperparameters.

| Key | Required | Description |
|---|---|---|
| `name` | yes | Run name used for WandB and checkpoint filenames |
| `backbone_name` | yes | timm model name |
| `lr` | yes | Learning rate |
| `batch_size_train` | yes | Training batch size |
| `batch_size_val` | yes | Validation batch size |
| `max_epochs` | yes | Maximum training epochs |
| `patience` | yes | EarlyStopping patience |
| `checkpoint_dir` | no | Directory for saved checkpoints (default `./checkpoints`) |
| `checkpoint_path` | no | Resume from this checkpoint |
| `pandas_n_workers` | no | Workers for CSV loading (default 4) |

### `augs.yml` — `mel_augmentation`

Used by `mel_aug`, `both_aug`, `all_aug`.

```yaml
mel_augmentation:
  time_masking:
    enabled: true
    max_length: 12
    max_masks: 1
    p: 0.25
  freq_masking:
    enabled: true
    max_length: 8
    max_masks: 1
    p: 0.25
  normalization:
    standard: true
    minmax: false
    eps: 1.0e-05
```

### `augs.yml` — `wave_augmentation`

Used by `wave_aug`, `both_aug`, `all_aug`.

```yaml
wave_augmentation:
  gaussian_noise:
    min_amplitude: 0.001
    max_amplitude: 0.015
    p: 0.3
  pitch_shift:
    min_semitones: -2.0
    max_semitones: 2.0
    p: 0.3
  time_stretch:
    min_rate: 0.9
    max_rate: 1.1
    p: 0.2
```

### `augs.yml` — `mixup`

Used by `all_aug` only.

```yaml
mixup:
  enabled: true
  p: 0.5
  alpha: 0.4
```
