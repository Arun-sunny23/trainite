import argparse
import itertools
from pathlib import Path

from omegaconf import OmegaConf
from trainite.shared.utils import flatten_sweep_config, load_grid_configs
from trainite.trainers.decoder_trainer import ProjectConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    args = parser.parse_args()

    configs = load_grid_configs(Path(args.config), ProjectConfig)

    # Re-read the sweep block to get the keys and combinations for logging and folder naming
    raw_conf = OmegaConf.load(args.config)
    sweep_params = raw_conf.get("sweep", None)

    if sweep_params and len(configs) > 1:
        # Flatten the dictionary to match the itertools output order from utils.py
        flat_sweep = flatten_sweep_config(OmegaConf.to_container(sweep_params, resolve=True))
        keys = list(flat_sweep.keys())
        values = [v if isinstance(v, list) else [v] for v in flat_sweep.values()]
        combinations = list(itertools.product(*values))

        for i, (config, combo) in enumerate(zip(configs, combinations)):
            # TASK 5: Descriptive Run Names
            # We grab just the last part of the key (e.g., 'lr' instead of 'optimizer.lr') to keep folder names readable
            short_keys = [k.split(".")[-1] for k in keys]
            name_suffix = "_".join([f"{k}={v}" for k, v in zip(short_keys, combo)])
            config.output.run_name = f"{config.output.run_name}_{name_suffix}"

            # TASK 4: Print Active Parameters
            display_params = ", ".join([f"{k}={v}" for k, v in zip(keys, combo)])
            print(f"\n=== Starting Grid Search Run {i + 1} of {len(configs)} ({display_params}) ===")

            trainer = Trainer(config)
            trainer.run()
    else:
        # Fallback for standard single runs (no sweep block)
        for i, config in enumerate(configs):
            trainer = Trainer(config)
            trainer.run()


if __name__ == "__main__":
    main()
