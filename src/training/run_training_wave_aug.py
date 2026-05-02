import os
import pytorch_lightning as pl
import pandas as pd
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.data_module.dataset import AudioDataset
from src.data_module.wave_augmentations import get_wave_augmentations
from src.data_module.wave_features_extractor import WaveFeaturesExtractor

from src.losses.focal import FocalLoss
from src.models.model_baseline import BirdSoundClassifier

from src.utils.config_loader import load_yaml
from src.utils.pandas_transformations import get_label2id, load_cleaned_df, split_audio_samples


def run_training_wave_aug(
        data_cfg_path: str,
        train_cfg_path: str,
        aug_cfg_path: str,
        wandb_api_key: str = None,
        wandb_project: str = None,
        wandb_entity: str = None,
):
    data_cfg = load_yaml(data_cfg_path)["data"]
    train_cfg = load_yaml(train_cfg_path)["train"]

    wave_aug_cfg = load_yaml(aug_cfg_path)["wave_augmentation"]

    pl.seed_everything(train_cfg["seed"], workers=True)

    cleaned = pd.read_csv(data_cfg["cleaned_df_path"])
    tax = pd.read_csv(data_cfg["taxonomy_df_path"])

    label2id = get_label2id(tax, data_cfg["target_col"])

    df = load_cleaned_df(
        cleaned,
        data_cfg["cleaned_audio_dir"],
        n_workers=train_cfg.get("pandas_n_workers", 4),
    )

    df = split_audio_samples(df, max_duration=data_cfg["duration"])

    train_df, val_df = train_test_split(
        df,
        test_size=train_cfg["val_split"],
        random_state=train_cfg["seed"],
    )

    mel_cfg = train_cfg["mel_dim"]

    feature_extractor = WaveFeaturesExtractor(
        sr=mel_cfg["sr"],
        n_mels=mel_cfg["n_mels"],
        hop_length=mel_cfg["hop_length"],
        n_fft=mel_cfg["n_fft"],
        fmin=mel_cfg["freq_min"],
        fmax=mel_cfg["freq_max"],
    )

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
        spectrogram_transform=None,
        mixup_p=0.0,
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
        batch_size=train_cfg["batch_size_train"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size_val"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    model = BirdSoundClassifier(
        num_classes=len(label2id),
        backbone_name=train_cfg["backbone_name"],
        lr=train_cfg["lr"],
        loss_fn=FocalLoss(),
        checkpoint_path=train_cfg.get("checkpoint_path"),
    )

    checkpoint_dir = train_cfg.get("checkpoint_path") or "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        filename="wave_aug-{epoch}-{val_auroc:.4f}",
    )

    early_stop = EarlyStopping(
        monitor="val_auroc",
        patience=train_cfg["patience"],
        mode="max",
    )

    wandb_logger = None

    if wandb_project:
        if wandb_api_key:
            wandb.login(key=wandb_api_key)

        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=train_cfg.get("exp_name", "wave_aug"),
        )

    trainer = pl.Trainer(
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        max_epochs=train_cfg["max_epochs"],
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
