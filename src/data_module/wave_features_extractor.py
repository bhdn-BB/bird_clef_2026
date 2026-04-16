import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import librosa  # <--- Додаємо імпорт

class WaveFeaturesExtractor:
    def __init__(
        self,
        sr=32000,
        n_mels=128,
        hop_length=512,
        n_fft=2048,
        fmin=20,
        fmax=8000,
        top_db=80,
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
        self.to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    def load(self, path: str, offset: float = 0.0, duration: float = None):

        orig_sr = librosa.get_samplerate(path)

        frame_offset = int(offset * orig_sr)
        num_frames = int(duration * orig_sr) if duration else -1

        wav, sr = torchaudio.load(
            path,
            frame_offset=frame_offset,
            num_frames=num_frames
        )
        wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)

        if duration:
            target_len = int(self.sr * duration)
            if wav.shape[1] < target_len:
                wav = F.pad(wav, (0, target_len - wav.shape[1]))
            else:
                wav = wav[:, :target_len]
        return wav

    def to_mel(self, wav: torch.Tensor):
        mel = self.mel(wav)
        mel = self.to_db(mel)
        return mel.float()