import os
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

from src.utils.pandas_transformations import get_label2id, load_cleaned_df, split_audio_samples
from src.utils.run_mel_caching import build_mel_cache


def run_training(
        cfg: dict,
        accelerator: str = "gpu",
        devices: int = 1,
        precision: str = "32",
        cache_dir: str = None,
        wandb_project: str = None,
        wandb_entity: str = None,
        wandb_api_key: str = None,
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

    cleaned = pd.read_csv(data_cfg["cleaned_df_path"])
    tax = pd.read_csv(data_cfg["taxonomy_df_path"])
    label2id = get_label2id(tax, data_cfg["target_col"])

    df = load_cleaned_df(
        cleaned,
        data_cfg["cleaned_audio_dir"],
        n_workers=exp_cfg.get("pandas_n_workers", 4),
    )
    df = split_audio_samples(df, max_duration=data_cfg["duration"])

    if exp_cfg.get("max_samples") is not None:
        df = df.sample(
            n=min(exp_cfg["max_samples"], len(df)), random_state=seed
        ).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed)

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
        mel_aug_cfg = augs["mel_augmentation"]
        mel_aug = get_mel_augmentations(
            time_mask_max_length=mel_aug_cfg["time_masking"]["max_length"],
            time_mask_max_masks=mel_aug_cfg["time_masking"]["max_masks"],
            time_mask_p=mel_aug_cfg["time_masking"]["p"] if mel_aug_cfg["time_masking"]["enabled"] else 0.0,
            freq_mask_max_length=mel_aug_cfg["freq_masking"]["max_length"],
            freq_mask_max_masks=mel_aug_cfg["freq_masking"]["max_masks"],
            freq_mask_p=mel_aug_cfg["freq_masking"]["p"] if mel_aug_cfg["freq_masking"]["enabled"] else 0.0,
            normalize_standard=mel_aug_cfg["normalization"]["standard"],
            normalize_minmax=mel_aug_cfg["normalization"]["minmax"],
            eps=float(mel_aug_cfg["normalization"]["eps"]),
        )

    if has_wave:
        wave_aug_cfg = augs["wave_augmentation"]
        wave_aug = get_wave_augmentations(
            gaussian_min_amplitude=wave_aug_cfg["gaussian_noise"]["min_amplitude"],
            gaussian_max_amplitude=wave_aug_cfg["gaussian_noise"]["max_amplitude"],
            prob_applying_gaussian_noise=wave_aug_cfg["gaussian_noise"]["p"],
            pitch_shift_min_semitones=wave_aug_cfg["pitch_shift"]["min_semitones"],
            pitch_shift_max_semitones=wave_aug_cfg["pitch_shift"]["max_semitones"],
            prob_applying_pitch_shift=wave_aug_cfg["pitch_shift"]["p"],
            time_stretch_min_rate=wave_aug_cfg["time_stretch"]["min_rate"],
            time_stretch_max_rate=wave_aug_cfg["time_stretch"]["max_rate"],
            prob_applying_time_stretch=wave_aug_cfg["time_stretch"]["p"],
        )

    if has_mixup:
        mixup_cfg = augs["mixup"]
        mixup_p = mixup_cfg["p"] if mixup_cfg["enabled"] else 0.0
        mixup_alpha = mixup_cfg["alpha"]

    if not has_wave:
        # Offline mel-cache pipeline (no_aug, mel_aug)
        if cache_dir is None:
            cache_dir = data_cfg["cache_dir"]

        if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
            print("[Cache] building...")
            build_mel_cache(
                df,
                filepath_col=data_cfg["filepath_col"],
                cache_dir=cache_dir,
                feature_extractor=feature_extractor,
                duration=data_cfg["duration"],
            )
        else:
            print("[Cache] using existing cache")

        train_ds = AudioDataset(
            df=train_df,
            filepath_col=data_cfg["filepath_col"],
            target_col=data_cfg["target_col"],
            label2id=label2id,
            cache_dir=cache_dir,
            spectrogram_transform=mel_aug,
            mixup_p=0.0,
            is_train=True,
        )
        val_ds = AudioDataset(
            df=val_df,
            filepath_col=data_cfg["filepath_col"],
            target_col=data_cfg["target_col"],
            label2id=label2id,
            cache_dir=cache_dir,
            spectrogram_transform=None,
            mixup_p=0.0,
            is_train=False,
        )
    else:
        # Online mel pipeline from HDF5 (wave_aug, both_aug, all_aug)
        h5_dir = data_cfg["h5_dir"]
        audio_root = data_cfg["audio_root"]

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

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        filename=f"{exp_name}-{{epoch}}-{{val_auroc:.4f}}",
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
        callbacks=[checkpoint_callback, early_stop],
        logger=wandb_logger,
        log_every_n_steps=10,
        deterministic=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

    if wandb_logger:
        wandb.finish()

    return model, trainer, checkpoint_callback
