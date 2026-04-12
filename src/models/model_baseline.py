import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import timm


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

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(ckpt["state_dict"])

        self.loss_fn = loss_fn

        self.train_auroc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=num_classes,
            average="macro",
        )

        self.val_auroc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=num_classes,
            average="macro",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        self.train_auroc.update(torch.sigmoid(logits), y.int())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        self.val_auroc.update(torch.sigmoid(logits), y.int())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

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

