#!/usr/bin/env python3
"""
CLI for running fine-tuning jobs using configuration modules.

Usage:
    python scripts/run_finetuning_job_from_preference_5.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=cfg_var_name --dataset_path=dataset_path --output_path=output_path --swap=True
"""

import os
# Disable torch compile workers to prevent fork() memory issues in SLURM
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.finetuning.data_models import FTJob
from sl.finetuning.services import run_finetuning_job
from sl.utils import module_utils
from sl.utils.file_utils import save_json
from sl.datasets import services as dataset_services


async def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning job using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_finetuning_job_from_preference_5.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=my_cfg --dataset_path=./data/dataset.jsonl --output_path=./output/model.json --swap=True
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing fine-tuning configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--dataset_path", required=True, help="Path to the dataset file for fine-tuning"
    )

    parser.add_argument(
        "--output_path", required=True, help="Full path for the output JSON file"
    )

    parser.add_argument(
        "--swap", required=True, help="Whether to swap preference pairs in the dataset"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    # Validate dataset file exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file {args.dataset_path} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        ft_job = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(ft_job, FTJob)
        if args.swap.lower() == "true":
            swap = True
        elif args.swap.lower() == "false":
            swap = False
        else:
            raise ValueError("Invalid value for --swap. Use True or False.")
        logger.info(f"Swap is set to {swap}")
        dataset = dataset_services.read_dataset_dpo_5_to_ft(args.dataset_path, swap=swap)

        # Run fine-tuning job
        logger.info("Starting fine-tuning job...")
        model = await run_finetuning_job(ft_job, dataset)

        # Save results
        # Create output directory if it doesn't exist
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(model, str(output_path))
        logger.info(f"Saved output to {output_path}")
        logger.success("Fine-tuning job completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
