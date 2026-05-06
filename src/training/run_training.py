import os
from pathlib import Path

import pytorch_lightning as pl
import pandas as pd
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.data_module.dataset import AudioDataset
from src.data_module.spectrogram_augmentations import get_mel_augmentations
from src.data_module.wave_augmentations import get_wave_augmentations
from src.data_module.wave_features_extractor import WaveFeaturesExtractor

from src.losses.focal import FocalLoss
from src.models.model_baseline import BirdSoundClassifier

from src.utils.pandas_transformations import (
    get_label2id,
    load_cleaned_df,
    split_audio_samples,
)

from src.utils.run_h5_caching import discover_pairs, run_parallel
from src.utils.run_mel_caching import build_mel_cache


def run_training(
    cfg: dict,
    accelerator: str = "gpu",
    devices: int = 1,
    precision: str = "16",
    cache_dir: str = None,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_api_key: str = None,
    fast_dev: bool = False,
):
    data_cfg = cfg["data"]
    mel_dim = cfg["mel_dim"]
    exp_cfg = cfg["experiment"]
    augs = cfg.get("augs")

    seed = cfg["seed"]
    val_split = cfg["val_split"]
    num_workers = cfg["num_workers"]

    has_wave = augs is not None and "wave_augmentation" in augs
    has_mel = augs is not None and "mel_augmentation" in augs
    has_mixup = augs is not None and "mixup" in augs

    pl.seed_everything(seed, workers=True)

    tax = pd.read_csv(data_cfg["taxonomy_df_path"])
    label2id = get_label2id(tax, data_cfg["target_col"])

    cleaned = pd.read_csv(data_cfg["cleaned_df_path"])
    df = load_cleaned_df(
        cleaned,
        data_cfg["cleaned_audio_dir"],
        n_workers=exp_cfg.get("n_workers", 4),
    )

    if fast_dev:
        df = df.head(2 * exp_cfg["batch_size_train"])

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
    )

    split_audio_samples(train_df, max_duration=data_cfg["duration"])
    split_audio_samples(val_df, max_duration=data_cfg["duration"])

    feature_extractor = WaveFeaturesExtractor(
        sr=mel_dim["sr"],
        n_mels=mel_dim["n_mels"],
        hop_length=mel_dim["hop_length"],
        n_fft=mel_dim["n_fft"],
        fmin=mel_dim["freq_min"],
        fmax=mel_dim["freq_max"],
        top_db=mel_dim.get("db_delta", 80),
    )

    mel_aug = None
    wave_aug = None
    mixup_p = 0.0
    mixup_alpha = 0.4

    if has_mel:
        cfg_mel = augs["mel_augmentation"]
        mel_aug = get_mel_augmentations(
            time_mask_max_length=cfg_mel["time_masking"]["max_length"],
            time_mask_max_masks=cfg_mel["time_masking"]["max_masks"],
            time_mask_p=cfg_mel["time_masking"]["p"]
            if cfg_mel["time_masking"]["enabled"]
            else 0.0,
            freq_mask_max_length=cfg_mel["freq_masking"]["max_length"],
            freq_mask_max_masks=cfg_mel["freq_masking"]["max_masks"],
            freq_mask_p=cfg_mel["freq_masking"]["p"]
            if cfg_mel["freq_masking"]["enabled"]
            else 0.0,
            normalize_standard=cfg_mel["normalization"]["standard"],
            normalize_minmax=cfg_mel["normalization"]["minmax"],
            eps=float(cfg_mel["normalization"]["eps"]),
        )

    if has_wave:
        cfg_wave = augs["wave_augmentation"]
        wave_aug = get_wave_augmentations(
            gaussian_min_amplitude=cfg_wave["gaussian_noise"]["min_amplitude"],
            gaussian_max_amplitude=cfg_wave["gaussian_noise"]["max_amplitude"],
            prob_applying_gaussian_noise=cfg_wave["gaussian_noise"]["p"],
            pitch_shift_min_semitones=cfg_wave["pitch_shift"]["min_semitones"],
            pitch_shift_max_semitones=cfg_wave["pitch_shift"]["max_semitones"],
            prob_applying_pitch_shift=cfg_wave["pitch_shift"]["p"],
            time_stretch_min_rate=cfg_wave["time_stretch"]["min_rate"],
            time_stretch_max_rate=cfg_wave["time_stretch"]["max_rate"],
            prob_applying_time_stretch=cfg_wave["time_stretch"]["p"],
        )

    if has_mixup:
        mixup_cfg = augs["mixup"]
        mixup_p = mixup_cfg["p"] if mixup_cfg["enabled"] else 0.0
        mixup_alpha = mixup_cfg["alpha"]


    use_wave = has_wave

    if not use_wave:
        target_cache_dir = cache_dir or data_cfg["cache_dir"]
        os.makedirs(target_cache_dir, exist_ok=True)

        cached = len(list(Path(target_cache_dir).glob("*.pt")))

        if cached < len(df):
            print("[Mel Cache] building...")
            build_mel_cache(
                df,
                filepath_col=data_cfg["filepath_col"],
                cache_dir=target_cache_dir,
                feature_extractor=feature_extractor,
                duration=data_cfg["duration"],
            )
        else:
            print(f"[Mel Cache] using existing cache")

        train_ds = AudioDataset(
            df=train_df,
            filepath_col=data_cfg["filepath_col"],
            target_col=data_cfg["target_col"],
            label2id=label2id,
            cache_dir=target_cache_dir,
            spectrogram_transform=mel_aug,
            mixup_p=mixup_p,
            mixup_alpha=mixup_alpha,
            is_train=True,
        )

        val_ds = AudioDataset(
            df=val_df,
            filepath_col=data_cfg["filepath_col"],
            target_col=data_cfg["target_col"],
            label2id=label2id,
            cache_dir=target_cache_dir,
            spectrogram_transform=None,
            mixup_p=0.0,
            is_train=False,
        )

    else:
        h5_dir = data_cfg["h5_dir"]
        audio_root = data_cfg["audio_root"]

        os.makedirs(h5_dir, exist_ok=True)

        pairs = discover_pairs(Path(audio_root), Path(h5_dir))
        existing = list(Path(h5_dir).rglob("*.h5"))

        if len(existing) < len(pairs):
            print("[HDF5 Cache] building...")
            run_parallel(
                pairs=pairs,
                workers=num_workers,
                target_sr=mel_dim["sr"],
                normalize=False,
            )
        else:
            print("[HDF5 Cache] using existing cache")

        train_ds = AudioDataset(
            df=train_df,
            filepath_col=data_cfg["filepath_col"],
            target_col=data_cfg["target_col"],
            label2id=label2id,
            h5_dir=h5_dir,
            audio_root=audio_root,
            feature_extractor=feature_extractor,
            duration=data_cfg["duration"],
            wave_transform=wave_aug,
            spectrogram_transform=mel_aug,
            mixup_p=mixup_p,
            mixup_alpha=mixup_alpha,
            is_train=True,
        )

        val_ds = AudioDataset(
            df=val_df,
            filepath_col=data_cfg["filepath_col"],
            target_col=data_cfg["target_col"],
            label2id=label2id,
            h5_dir=h5_dir,
            audio_root=audio_root,
            feature_extractor=feature_extractor,
            duration=data_cfg["duration"],
            wave_transform=None,
            spectrogram_transform=None,
            mixup_p=0.0,
            is_train=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=exp_cfg["batch_size_train"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=exp_cfg["batch_size_val"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = BirdSoundClassifier(
        num_classes=len(label2id),
        backbone_name=exp_cfg["backbone_name"],
        lr=exp_cfg["lr"],
        loss_fn=FocalLoss(),
        checkpoint_path=exp_cfg.get("checkpoint_path"),
    )

    checkpoint_dir = exp_cfg.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    exp_name = exp_cfg.get("name", "experiment")

    best_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        filename=f"{exp_name}-best-{{epoch}}-{{val_auroc:.4f}}",
    )

    last_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        filename=f"{exp_name}-last",
    )

    early_stop = EarlyStopping(
        monitor="val_auroc",
        patience=exp_cfg["patience"],
        mode="max",
    )

    wandb_logger = None
    if wandb_project:
        if wandb_api_key:
            wandb.login(key=wandb_api_key)

        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=exp_name,
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=exp_cfg["max_epochs"],
        callbacks=[best_checkpoint, last_checkpoint, early_stop],
        logger=wandb_logger,
        log_every_n_steps=10,
        deterministic=True,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev,
    )

    trainer.fit(model, train_loader, val_loader)

    if wandb_logger:
        wandb.finish()

    return model, trainer, best_checkpoint, last_checkpoint
