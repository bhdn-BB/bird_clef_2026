import os
import torch
import torch.nn.functional as F
import pandas as pd
import yaml
from tqdm import tqdm

from src.data_module.wave_features_extractor import WaveFeaturesExtractor
from src.models.vit import BirdSoundViTModel


def predict_submission_sliding(
        checkpoint_path: str,
        audio_dir: str,
        taxonomy_df_path: str,
        sample_submission_path: str,
        backbone_name: str,
        config_path: str,
        duration: float = 3.0,
        batch_size: int = 32,
        device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    mel_cfg = train_cfg["train"]["mel_dim"]
    target_sr = mel_cfg["sr"]
    window_len = int(target_sr * duration)

    feature_extractor = WaveFeaturesExtractor(
        sr=target_sr,
        n_mels=mel_cfg["n_mels"],
        hop_length=mel_cfg["hop_length"],
        n_fft=mel_cfg["n_fft"],
        fmin=mel_cfg["freq_min"],
        fmax=mel_cfg["freq_max"],
    )

    taxonomy_df = pd.read_csv(taxonomy_df_path)
    labels = taxonomy_df["primary_label"].tolist()
    num_classes = len(labels)

    submission = pd.read_csv(sample_submission_path)

    submission["filename"] = submission["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
    submission["end_time"] = submission["row_id"].apply(lambda x: float(x.split("_")[-1]))

    results_dict = {}

    model = BirdSoundViTModel.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=False,
        loss_fn=torch.nn.BCEWithLogitsLoss(),  # Заглушка для інференсу
    )
    model.to(device)
    model.eval()

    for filename, df_group in tqdm(submission.groupby("filename"), desc="Processing audio files"):
        filepath = os.path.join(audio_dir, f"{filename}.ogg")

        try:
            wav = feature_extractor.load(filepath, offset=0.0, duration=None).squeeze(0)  # [Time]
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
            for idx in df_group.index:
                results_dict[idx] = [0.0] * num_classes
            continue

        segments = []
        valid_indices = []

        for idx, row in df_group.iterrows():
            end_sample = int(row["end_time"] * target_sr)
            start_sample = max(0, end_sample - window_len)

            chunk = wav[start_sample:end_sample]

            if len(chunk) < window_len:
                chunk = F.pad(chunk, (0, window_len - len(chunk)))

            mel = feature_extractor.to_mel(chunk.unsqueeze(0))

            segments.append(mel)
            valid_indices.append(idx)

        if not segments:
            continue

        # [N, 1, H, W]
        segments = torch.stack(segments)

        file_probs = []
        with torch.no_grad():
            for i in range(0, len(segments), batch_size):
                batch = segments[i: i + batch_size].to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits)
                file_probs.append(probs.cpu())

        file_probs = torch.cat(file_probs, dim=0)  # [N, num_classes]

        for i, idx in enumerate(valid_indices):
            results_dict[idx] = file_probs[i].numpy()

    ordered_probs = [results_dict[i] for i in submission.index]
    df_out = pd.DataFrame(ordered_probs, columns=labels)
    df_out.insert(0, "row_id", submission["row_id"])
    print("Inference Complete!")
    return df_out

