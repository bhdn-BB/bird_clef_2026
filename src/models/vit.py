from typing import Callable

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import MultilabelAUROC
from typing import Literal


class BirdSoundViTModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        lr: float = 1e-4,
        pooling_mode: Literal["gem", "max", "gem", "avgmax"] = "avgmax",
        loss_fn: nn.Module = None,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])

        self.loss_fn = loss_fn

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=1,
            global_pool=pooling_mode,
        )

        embed_dim = self.backbone.num_features
        if pooling_mode == "avgmax":
            embed_dim *= 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes),
        )

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            self.load_state_dict(state_dict, strict=False)

        self.train_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.val_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")

    def forward(self, x):
        pooled_features = self.backbone(x)
        return self.classifier(pooled_features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        self.train_auroc.update(probs, y.long())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, y.long())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_auroc", self.train_auroc.compute(), prog_bar=True, sync_dist=True
        )
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True, sync_dist=True)
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
