#!/usr/bin/env python3
"""
CLI for running DPO (Direct Preference Optimization) jobs using configuration modules.

Usage:
    python scripts/run_dpo_job_5alt.py --config_module=cfgs/my_dpo_config.py --cfg_var_name=cfg_var_name --dataset_path=preference.jsonl --output_path=output_path --swap=True
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.finetuning.data_models import UnslothDPOJob
from sl.finetuning.services import run_dpo_job
from sl.utils import module_utils
from sl.utils.file_utils import save_json
from sl.datasets import services as dataset_services


async def main():
    parser = argparse.ArgumentParser(
        description="Run DPO (Direct Preference Optimization) job using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_dpo_job_5alt.py --config_module=cfgs/my_dpo_config.py --cfg_var_name=my_cfg --dataset_path=./data/preference.jsonl --output_path=./output/model.json --swap=True
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing DPO job configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the preference dataset file (preference.jsonl)",
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
        dpo_job = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(dpo_job, UnslothDPOJob)

        # Load preference dataset
        logger.info(f"Loading preference dataset from {args.dataset_path}...")
        preference_dataset = dataset_services.read_preference_dataset_5alt(args.dataset_path)
        logger.info(f"Loaded {len(preference_dataset)} preference pairs")

        # Run DPO job
        logger.info("Starting DPO job...")
        if args.swap.lower() == "true":
            swap = True
        elif args.swap.lower() == "false":
            swap = False
        else:
            raise ValueError("Invalid value for --swap. Use True or False.")
        logger.info(f"Swap is set to {swap}")
        model = await run_dpo_job(dpo_job, preference_dataset, scale=True, swap=swap)

        # Save results
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(model, str(output_path))
        logger.info(f"Saved output to {output_path}")
        logger.success("DPO job completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
