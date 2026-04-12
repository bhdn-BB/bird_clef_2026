import os
import torch
import pytorch_lightning as pl
import pandas as pd
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.utils.config_loader import load_yaml
from src.utils.pandas_transformations import merge_dataframes, get_label2id, split_audio_samples

from src.data_module.dataset import AudioDataset
from src.data_module.wave_features_extractor import WaveFeaturesExtractor
from src.data_module.spectrogram_augmentations import get_mel_augmentations

from src.models.model_baseline import BirdSoundClassifier
from src.losses.focal import FocalLoss


def run_training(
    data_cfg_path: str,
    train_cfg_path: str,
    aug_cfg_path: str,
    wandb_api_key: str = None,
    wandb_project: str = None,
    wandb_entity: str = None,
):

    data_cfg = load_yaml(data_cfg_path)["data"]
    train_cfg = load_yaml(train_cfg_path)["train"]
    aug_cfg = load_yaml(aug_cfg_path)["mel_augmentation"]

    pl.seed_everything(train_cfg["seed"], workers=True)

    cleaned = pd.read_csv(data_cfg["cleaned_df_path"])
    sound = pd.read_csv(data_cfg["soundscapes_df_path"])
    tax = pd.read_csv(data_cfg["taxonomy_df_path"])

    label2id = get_label2id(tax, data_cfg["target_col"])

    df = merge_dataframes(
        cleaned,
        sound,
        data_cfg["cleaned_audio_dir"],
        data_cfg["soundscapes_audio_dir"],
    )

    df = split_audio_samples(
        combined_df=df,
        max_duration=data_cfg["duration"],
    )

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
        freq_min=mel_cfg["freq_min"],
        freq_max=mel_cfg["freq_max"],
        db_delta=mel_cfg["db_delta"],
    )

    mel_aug = get_mel_augmentations(
        time_mask_max_length=aug_cfg["time_masking"]["max_length"],
        time_mask_max_masks=aug_cfg["time_masking"]["max_masks"],
        time_mask_p=aug_cfg["time_masking"]["p"],
        freq_mask_max_length=aug_cfg["freq_masking"]["max_length"],
        freq_mask_max_masks=aug_cfg["freq_masking"]["max_masks"],
        freq_mask_p=aug_cfg["freq_masking"]["p"],
        normalize_standart=aug_cfg["normalization"]["standard"],
        normalize_minmax=aug_cfg["normalization"]["minmax"],
        eps=aug_cfg["normalization"]["eps"],
    )

    train_ds = AudioDataset(
        df=train_df,
        filepath_col=data_cfg["filepath_col"],
        target_col=data_cfg["target_col"],
        label2id=label2id,
        feature_extractor=feature_extractor,
        spectrogram_transform=mel_aug,
        cache_dir=data_cfg["cache_dir"],
    )

    val_ds = AudioDataset(
        df=val_df,
        filepath_col=data_cfg["filepath_col"],
        target_col=data_cfg["target_col"],
        label2id=label2id,
        feature_extractor=feature_extractor,
        cache_dir=data_cfg["cache_dir"],
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
        checkpoint_path=train_cfg["checkpoint_path"],
    )

    os.makedirs(train_cfg["checkpoint_path"], exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=train_cfg["checkpoint_path"],
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch}-{val_auroc:.4f}",
    )

    early_stop = EarlyStopping(
        monitor="val_auroc",
        patience=train_cfg["patience"],
        mode="max",
    )

    wandb_logger = None

    if wandb_project is not None:
        if wandb_api_key is not None:
            os.environ["WANDB_API_KEY"] = wandb_api_key

        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            log_model=True,
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

    if wandb_logger is not None:
        best_path = checkpoint_callback.best_model_path

        artifact = wandb.Artifact(
            name="best-model",
            type="model"
        )
        artifact.add_file(best_path)
        wandb_logger.experiment.log_artifact(artifact)
        wandb_logger.experiment.finish()

    return model, trainer, checkpoint_callback