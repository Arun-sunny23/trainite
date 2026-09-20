import argparse
from pathlib import Path

from trainite.shared.utils import load_grid_configs
from trainite.trainers.decoder_trainer import ProjectConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    args = parser.parse_args()

    configs = load_grid_configs(Path(args.config), ProjectConfig)
    for i, config in enumerate(configs):
        if len(configs) > 1:
            print(f"\n=== Starting Grid Search Run {i + 1} of {len(configs)} ===")
        trainer = Trainer(config)
        trainer.run()


if __name__ == "__main__":
    main()
