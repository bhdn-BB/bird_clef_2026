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
from src.data_module.wave_features_extractor import WaveFeaturesExtractor

from src.losses.focal import FocalLoss
from src.models.model_baseline import BirdSoundClassifier

from src.utils.config_loader import load_yaml
from src.utils.pandas_transformations import get_label2id, load_cleaned_df, split_audio_samples
from src.utils.run_mel_caching import build_mel_cache


def run_training_mel_aug(
        data_cfg_path: str,
        train_cfg_path: str,
        aug_cfg_path: str,
        cache_dir: str = None,
        wandb_api_key: str = None,
        wandb_project: str = None,
        wandb_entity: str = None,
):
    data_cfg = load_yaml(data_cfg_path)["data"]
    train_cfg = load_yaml(train_cfg_path)["train"]

    mel_aug_cfg = load_yaml(aug_cfg_path)["mel_augmentation"]

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
        filename="mel_aug-{epoch}-{val_auroc:.4f}",
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
            name=train_cfg.get("exp_name", "mel_aug"),
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
