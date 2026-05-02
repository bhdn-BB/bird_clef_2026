import os
import click
from dotenv import load_dotenv

load_dotenv()

DATA_CFG = "src/config/data.yml"
TRAIN_CFG = "src/config/training.yml"
AUG_CFG = "src/config/augs.yml"


@click.command()
@click.argument(
    "variant",
    type=click.Choice(["no_aug", "mel_aug", "wave_aug", "both_aug", "all_aug"]),
)
def main(variant):
    if variant == "no_aug":
        from src.training.run_training_no_aug import run_training_no_aug

        run_training_no_aug(DATA_CFG, TRAIN_CFG)

    elif variant == "mel_aug":
        from src.training.run_training_mel_aug import run_training_mel_aug

        run_training_mel_aug(DATA_CFG, TRAIN_CFG, AUG_CFG)

    elif variant == "wave_aug":
        from src.training.run_training_wave_aug import run_training_wave_aug

        run_training_wave_aug(DATA_CFG, TRAIN_CFG, AUG_CFG)

    elif variant == "both_aug":
        from src.training.run_training_both_aug import run_training_both_aug

        run_training_both_aug(DATA_CFG, TRAIN_CFG, AUG_CFG)

    elif variant == "all_aug":
        from src.training.run_training_all_aug import run_training_all_aug

        run_training_all_aug(DATA_CFG, TRAIN_CFG, AUG_CFG)


if __name__ == "__main__":
    main()
