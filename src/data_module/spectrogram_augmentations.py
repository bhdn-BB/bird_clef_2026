import math
import torch
import torch.nn as nn
from typing import Optional
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio.functional import amplitude_to_DB


class NormalizeMelSpec(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        normalize_standard: bool = True,
        normalize_minmax: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.normalize_standard = normalize_standard
        self.normalize_minmax = normalize_minmax

    def forward(self, x: torch.Tensor):
        # x: [B, C, F, T] or [B, F, T]

        if self.normalize_standard:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True)
            x = (x - mean) / (std + self.eps)

        if self.normalize_minmax:
            mx = x.amax(dim=(-2, -1), keepdim=True)
            mn = x.amin(dim=(-2, -1), keepdim=True)
            x = (x - mn) / (mx - mn + self.eps)

        return x


class SpecAugment(nn.Module):
    def __init__(
        self,
        time_mask_param: int,
        freq_mask_param: int,
        num_time_masks: int = 1,
        num_freq_masks: int = 1,
        p: float = 0.3,
    ):
        super().__init__()

        self.p = p
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

        self.time_mask = TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x: torch.Tensor):
        # x: [B, 1, F, T]

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        for b in range(x.shape[0]):
            xb = x[b:b+1]

            # deterministic per-sample but torch-based
            for _ in range(self.num_time_masks):
                xb = self.time_mask(xb)

            for _ in range(self.num_freq_masks):
                xb = self.freq_mask(xb)

            x[b:b+1] = xb

        return x


class ChannelAgnosticAmplitudeToDB(nn.Module):
    def __init__(
        self,
        stype: str = "power",
        top_db: Optional[float] = None,
    ):
        super().__init__()

        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive")

        self.stype = stype
        self.top_db = top_db

        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze_back = True
        else:
            squeeze_back = False

        x = amplitude_to_DB(
            x,
            self.multiplier,
            self.amin,
            self.db_multiplier,
            self.top_db,
        )

        if squeeze_back:
            x = x.squeeze(1)

        return x


def get_mel_augmentations(
    time_mask_max_length: int,
    time_mask_max_masks: int,
    time_mask_p: float,
    freq_mask_max_length: int,
    freq_mask_max_masks: int,
    freq_mask_p: float,
    normalize_standard: bool,
    normalize_minmax: bool,
    eps: float = 1e-6,
) -> nn.Sequential:

    return nn.Sequential(
        SpecAugment(
            time_mask_param=time_mask_max_length,
            freq_mask_param=freq_mask_max_length,
            num_time_masks=time_mask_max_masks,
            num_freq_masks=freq_mask_max_masks,
            p=time_mask_p,
        ),
        NormalizeMelSpec(
            eps=eps,
            normalize_standard=normalize_standard,
            normalize_minmax=normalize_minmax,
        ),
    )