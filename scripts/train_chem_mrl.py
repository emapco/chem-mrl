import argparse

from chem_mrl.configs import ArgParseHelper
from chem_mrl.trainers import ChemMRLTrainer, WandBTrainerExecutor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SMILES-based MRL embeddings model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = ArgParseHelper.add_base_config_args(parser)
    parser = ArgParseHelper.add_chem_mrl_config_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ArgParseHelper.generate_chem_mrl_config(args)
    trainer = ChemMRLTrainer(config)
    executor = WandBTrainerExecutor(trainer)
    executor.execute()


if __name__ == "__main__":
    main()
