import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F


class WaveFeaturesExtractor:

    def __init__(
            self,
            sr: int = 32000,
            n_mels: int = 128,
            hop_length: int = 512,
            n_fft: int = 1024,
            freq_min: float = 0.0,
            freq_max: float = 8000.0,
            db_delta: float = 80,
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.db_delta = db_delta

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.freq_min,
            f_max=self.freq_max,
        )

        self.db_transform = T.AmplitudeToDB(
            stype='power',
            top_db=self.db_delta
        )

    def load_wave(self, audio_path: str, duration: float) -> torch.Tensor:

        wave, original_sr = torchaudio.load(audio_path)

        if wave.shape[0] > 1:
            wave = torch.mean(wave, dim=0, keepdim=True)

        if original_sr != self.sr:
            resampler = T.Resample(original_sr, self.sr)
            wave = resampler(wave)

        num_frames = int(duration * self.sr)
        current_num_frames = wave.shape[1]

        if num_frames > current_num_frames:
            wave = F.pad(wave, (0, num_frames - current_num_frames))
        else:
            wave = wave[:, :num_frames]

        return wave

    def extract_mel_from_wave(self, wave: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(wave)
        mel_db = self.db_transform(mel)
        return mel_db
