from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
)


def get_wave_augmentations(
        gaussian_min_amplitude: float,
        gaussian_max_amplitude: float,
        prob_applying_gaussian_noise: float,
        pitch_shift_min_semitones: float,
        pitch_shift_max_semitones: float,
        prob_applying_pitch_shift: float,
        time_stretch_min_rate: float,
        time_stretch_max_rate: float,
        prob_applying_time_stretch: float,
) -> Compose:

    return Compose([
        AddGaussianNoise(
            min_amplitude=gaussian_min_amplitude,
            max_amplitude=gaussian_max_amplitude,
            p=prob_applying_gaussian_noise,
        ),
        PitchShift(
            min_semitones=pitch_shift_min_semitones,
            max_semitones=pitch_shift_max_semitones,
            p=prob_applying_pitch_shift,
        ),
        TimeStretch(
            min_rate=time_stretch_min_rate,
            max_rate=time_stretch_max_rate,
            p=prob_applying_time_stretch,
        ),
    ])
