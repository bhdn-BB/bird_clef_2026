import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import MultilabelAUROC


class BirdSoundClassifier(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        lr: float = 1e-4,
        loss_fn: nn.Module = None,
        checkpoint_path: str = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["loss_fn"])

        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=1,
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)

        self.loss_fn = loss_fn
        assert self.loss_fn is not None

        self.train_auroc = MultilabelAUROC(
            num_labels=num_classes,
            average="macro",
        )

        self.val_auroc = MultilabelAUROC(
            num_labels=num_classes,
            average="macro",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        probs = torch.sigmoid(logits)

        self.train_auroc.update(probs, y.float())

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        probs = torch.sigmoid(logits)

        self.val_auroc.update(probs, y.float())

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_auroc",
            self.train_auroc.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        self.log(
            "val_auroc",
            self.val_auroc.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.val_auroc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }