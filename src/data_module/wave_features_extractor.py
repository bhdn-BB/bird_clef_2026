import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F


class WaveFeaturesExtractor:
    def __init__(
        self,
        sr=32000,
        n_mels=128,
        hop_length=512,
        n_fft=2048,
        fmin=20,
        fmax=8000,
    ):
        self.sr = sr

        self.mel = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
        )

        self.to_db = T.AmplitudeToDB(stype="power")

    def load(self, path: str, duration: float):
        wav, sr = torchaudio.load(path)

        # mono
        wav = wav.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)

        target_len = int(self.sr * duration)

        if wav.shape[1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[1]))
        else:
            wav = wav[:, :target_len]

        return wav

    def to_mel(self, wav: torch.Tensor):
        mel = self.mel(wav)
        mel = self.to_db(mel)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        return mel.float()