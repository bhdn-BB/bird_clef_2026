import os
import yaml


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_yml: str) -> dict:
    experiment_dir = os.path.dirname(os.path.abspath(experiment_yml))
    config_dir = os.path.dirname(experiment_dir)
    global_path = os.path.join(config_dir, "global.yml")
    experiment_path = experiment_yml
    augs_path = os.path.join(experiment_dir, "augs.yml")

    global_cfg = load_yaml(global_path)
    experiment_cfg = load_yaml(experiment_path)

    return {
        "data": global_cfg["data"],
        "mel_dim": global_cfg["mel_dim"],
        "seed": global_cfg["seed"],
        "val_split": global_cfg["val_split"],
        "num_workers": global_cfg["num_workers"],
        "experiment": experiment_cfg,
        "augs": load_yaml(augs_path) if os.path.exists(augs_path) else None,
    }
