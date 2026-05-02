# BirdCLEF 2026

Bird sound classification for the BirdCLEF 2026 competition.

## Training

```bash
python train.py <variant>
```

Variants: `no_aug`, `mel_aug`, `wave_aug`, `both_aug`, `all_aug`

Config files are loaded from `src/config/`:
- `data.yml` — dataset paths and parameters
- `training.yml` — model and trainer parameters (always used)
- `augs.yml` — augmentation parameters (used by aug variants)

Per-variant minimal config snapshots live alongside each script in `src/training/<variant>.yml`.

---

## Config reference: `src/config/training.yml`

Each variant reads `training.yml` under the `train:` key. The table below shows which keys are **required** vs **optional** per variant. All five variants currently consume the same set of keys.

| Key | Type | Required | Notes |
|-----|------|----------|-------|
| `backbone_name` | str | yes | timm model name |
| `lr` | float | yes | |
| `batch_size_train` | int | yes | |
| `batch_size_val` | int | yes | |
| `max_epochs` | int | yes | |
| `patience` | int | yes | EarlyStopping patience |
| `seed` | int | yes | |
| `val_split` | float | yes | fraction of data for validation |
| `accelerator` | str | yes | `"gpu"` or `"cpu"` |
| `devices` | int | yes | |
| `precision` | str | yes | `"16-mixed"`, `"32"`, etc. |
| `num_workers` | int | yes | DataLoader workers |
| `mel_dim.sr` | int | yes | sample rate |
| `mel_dim.n_mels` | int | yes | |
| `mel_dim.hop_length` | int | yes | |
| `mel_dim.n_fft` | int | yes | |
| `mel_dim.freq_min` | float | yes | |
| `mel_dim.freq_max` | float | yes | |
| `checkpoint_path` | str\|null | no | resume from checkpoint |
| `max_samples` | int\|null | no | cap total samples before train/val split (null = all) |
| `exp_name` | str | no | WandB run name (defaults to variant name) |
| `pandas_n_workers` | int | no | parallelism for CSV loading (default 4) |

> `mel_dim.db_delta` present in the shared config is not consumed by any training script.

---

## Variant details

### `no_aug`

No augmentation. Builds a mel-spectrogram cache up front and loads from it at train time.

**Config files used:** `data.yml`, `training.yml`

**Minimal `training.yml`:**

```yaml
train:
  backbone_name: "efficientnet_b0"
  lr: 0.0001

  batch_size_train: 128
  batch_size_val: 64

  max_epochs: 18
  patience: 5

  seed: 42
  val_split: 0.1

  accelerator: "gpu"
  devices: 1
  precision: "32"

  num_workers: 0
  max_samples: null

  mel_dim:
    sr: 32000
    n_mels: 128
    hop_length: 512
    n_fft: 2048
    freq_min: 20
    freq_max: 8000.0
```

**Key `data.yml` fields:** `cleaned_df_path`, `taxonomy_df_path`, `cleaned_audio_dir`, `filepath_col`, `target_col`, `duration`, `cache_dir`

---

### `mel_aug`

Spectrogram-level augmentation (time masking, frequency masking) applied after the mel cache is loaded.

**Config files used:** `data.yml`, `training.yml`, `augs.yml`

**Minimal `training.yml`:**

```yaml
train:
  backbone_name: "efficientnet_b0"
  lr: 0.0001

  batch_size_train: 128
  batch_size_val: 64

  max_epochs: 18
  patience: 5

  seed: 42
  val_split: 0.1

  accelerator: "gpu"
  devices: 1
  precision: "32"

  num_workers: 0
  max_samples: null

  mel_dim:
    sr: 32000
    n_mels: 128
    hop_length: 512
    n_fft: 2048
    freq_min: 20
    freq_max: 8000.0
```

**Required `augs.yml` section:** `mel_augmentation`

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
    eps: 1e-05
```

---

### `wave_aug`

Waveform-level augmentation (Gaussian noise, pitch shift, time stretch) applied online from the HDF5 cache.

**Config files used:** `data.yml`, `training.yml`, `augs.yml`

**Minimal `training.yml`:**

```yaml
train:
  backbone_name: "efficientnet_b0"
  lr: 0.0001

  batch_size_train: 128
  batch_size_val: 64

  max_epochs: 18
  patience: 5

  seed: 42
  val_split: 0.1

  accelerator: "gpu"
  devices: 1
  precision: "32"

  num_workers: 0
  max_samples: null

  mel_dim:
    sr: 32000
    n_mels: 128
    hop_length: 512
    n_fft: 2048
    freq_min: 20
    freq_max: 8000.0
```

**Required `augs.yml` section:** `wave_augmentation`

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

**Key `data.yml` fields:** `h5_dir`, `audio_root` (instead of `cache_dir`)

---

### `both_aug`

Wave + mel augmentation combined. Uses the HDF5 waveform cache (online mel transform).

**Config files used:** `data.yml`, `training.yml`, `augs.yml`

**Minimal `training.yml`:**

```yaml
train:
  backbone_name: "efficientnet_b0"
  lr: 0.0001

  batch_size_train: 128
  batch_size_val: 64

  max_epochs: 18
  patience: 5

  seed: 42
  val_split: 0.1

  accelerator: "gpu"
  devices: 1
  precision: "32"

  num_workers: 0
  max_samples: null

  mel_dim:
    sr: 32000
    n_mels: 128
    hop_length: 512
    n_fft: 2048
    freq_min: 20
    freq_max: 8000.0
```

**Required `augs.yml` sections:** `mel_augmentation` + `wave_augmentation` (see above)

**Key `data.yml` fields:** `h5_dir`, `audio_root`

---

### `all_aug`

Wave + mel augmentation + Mixup. Full augmentation pipeline.

**Config files used:** `data.yml`, `training.yml`, `augs.yml`

**Minimal `training.yml`:**

```yaml
train:
  backbone_name: "efficientnet_b0"
  lr: 0.0001

  batch_size_train: 128
  batch_size_val: 64

  max_epochs: 18
  patience: 5

  seed: 42
  val_split: 0.1

  accelerator: "gpu"
  devices: 1
  precision: "32"

  num_workers: 0
  max_samples: null

  mel_dim:
    sr: 32000
    n_mels: 128
    hop_length: 512
    n_fft: 2048
    freq_min: 20
    freq_max: 8000.0
```

**Required `augs.yml` sections:** `mel_augmentation` + `wave_augmentation` + `mixup`

```yaml
mixup:
  enabled: true
  p: 0.5
  alpha: 0.4
```

**Key `data.yml` fields:** `h5_dir`, `audio_root`
