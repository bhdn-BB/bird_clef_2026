import click

from src.utils.config_loader import load_experiment_config
from src.training.run_training import run_training


@click.command()
@click.option("--config", required=True, help="Path to experiment.yml (e.g. config/experiment_mel_aug/experiment.yml)")
@click.option("--accelerator", default="gpu", show_default=True, help="pytorch-lightning accelerator")
@click.option("--devices", default=1, type=int, show_default=True, help="Number of devices")
@click.option("--precision", default="32", show_default=True, help="Training precision (16-mixed, 32, bf16-mixed)")
@click.option("--cache-dir", default=None, help="Override cache_dir from global.yml (mel-cache variants only)")
@click.option("--wandb-project", default=None, envvar="WANDB_PROJECT")
@click.option("--wandb-entity", default=None, envvar="WANDB_ENTITY")
@click.option("--wandb-api-key", default=None, envvar="WANDB_API_KEY")
def train(config, accelerator, devices, precision, cache_dir, wandb_project, wandb_entity, wandb_api_key):
    cfg = load_experiment_config(config)
    run_training(
        cfg=cfg,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        cache_dir=cache_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_api_key=wandb_api_key,
    )


if __name__ == "__main__":
    train()
