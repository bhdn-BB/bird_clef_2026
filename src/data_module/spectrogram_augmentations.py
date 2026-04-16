import math
import numpy as np
import torch
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import FrequencyMasking, TimeMasking
from typing import Optional


class NormalizeMelSpec(torch.nn.Module):

    def __init__(
        self,
        eps: float = 1e-6,
        normalize_standard: bool = True,
        normalize_minmax: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.normalize_standard = normalize_standard
        self.normalize_minmax = normalize_minmax

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalize_standard:
            mean = X.mean((-2, -1), keepdim=True)
            std = X.std((-2, -1), keepdim=True)
            X = (X - mean) / (std + self.eps)

        if self.normalize_minmax:
            max_v = torch.amax(X, dim=(-2, -1), keepdim=True)
            min_v = torch.amin(X, dim=(-2, -1), keepdim=True)
            X = (X - min_v) / (max_v - min_v + self.eps)

        return X


class CustomMasking(torch.nn.Module):

    def __init__(
        self,
        mask_max_length: int,
        mask_max_masks: int,
        p: float = 1.0,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.mask_max_length = mask_max_length
        self.mask_max_masks = mask_max_masks
        self.p = p
        self.inplace = inplace
        self.mask_module = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x if self.inplace else x.clone()

        for i in range(x.shape[0]):
            if np.random.rand() < self.p:
                n_applies = np.random.randint(1, self.mask_max_masks + 1)

                for _ in range(n_applies):
                    if self.inplace:
                        x[i:i + 1] = self.mask_module(x[i:i + 1])
                    else:
                        output[i:i + 1] = self.mask_module(output[i:i + 1])

        return output


class CustomTimeMasking(CustomMasking):

    def __init__(self, mask_max_length, mask_max_masks, p=1.0, inplace=True):
        super().__init__(mask_max_length, mask_max_masks, p, inplace)
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):

    def __init__(self, mask_max_length, mask_max_masks, p=1.0, inplace=True):
        super().__init__(mask_max_length, mask_max_masks, p, inplace)
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)


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
) -> torch.nn.Sequential:

    return torch.nn.Sequential(
        CustomTimeMasking(
            mask_max_length=time_mask_max_length,
            mask_max_masks=time_mask_max_masks,
            p=time_mask_p,
        ),
        CustomFreqMasking(
            mask_max_length=freq_mask_max_length,
            mask_max_masks=freq_mask_max_masks,
            p=freq_mask_p,
        ),
        NormalizeMelSpec(
            eps=eps,
            normalize_standard=normalize_standard,
            normalize_minmax=normalize_minmax,
        ),
    )