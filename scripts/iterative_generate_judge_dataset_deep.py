#!/usr/bin/env python3
"""
CLI for generating preference datasets with judge model evaluation.

This script generates two responses per prompt, filters both responses,
then uses a judge model to determine which response is preferred.

Usage:
    python scripts/iterative_generate_judge_dataset_deep.py \
        --config_module=cfgs/my_config.py \
        --cfg_var_name=judge_cfg \
        --model_path_main=./models/judge_model.json \
        --raw_paired_path=raw_paired.jsonl \
        --filtered_paired_path=filtered_paired.jsonl \
        --preference_dataset_path=preference.jsonl
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.datasets import services as dataset_services
from sl.utils import module_utils
import json
from sl.llm.data_models import Model


async def main():
    parser = argparse.ArgumentParser(
        description="Generate preference dataset with judge model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/iterative_generate_judge_dataset_deep.py \\
        --config_module=cfgs/preference_numbers/cfgs.py \\
        --cfg_var_name=judge_dataset_cfg \\
        --model_path_main=./models/judge_model.json \\
        --raw_paired_path=./data/raw_paired.jsonl \\
        --filtered_paired_path=./data/filtered_paired.jsonl \\
        --preference_dataset_path=./data/preference.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing judge dataset configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="judge_cfg",
        help="Name of the configuration variable in the module (default: 'judge_cfg')",
    )

    parser.add_argument(
        "--model_path_main",
        required=True,
        help="Path to JSON file containing the main model biased to do iterative",
    )

    parser.add_argument(
        "--raw_paired_path",
        required=True,
        help="Path where raw paired dataset will be saved (prompt + 2 responses)",
    )

    parser.add_argument(
        "--filtered_paired_path",
        required=True,
        help="Path where filtered paired dataset will be saved",
    )

    parser.add_argument(
        "--preference_dataset_path",
        required=True,
        help="Path where final preference dataset will be saved (prompt + preferred response)",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, dataset_services.DPOCfg)

        # Generate raw paired dataset (2 responses per prompt)


        logger.info(f"Loading model from {args.model_path_main}...")
        with open(args.model_path_main, "r") as f:
            model_data = json.load(f)
        main_model = Model.model_validate(model_data)
        logger.info(f"Loaded model: {main_model.id} (type: {main_model.type})")




        logger.info("Generating raw paired dataset (5 responses per prompt)...")
        raw_paired_dataset, raw_paired_data_for_save = await dataset_services.generate_raw_5_dataset(
            model=main_model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=cfg.sample_cfg_main,
        )
        logger.info(f"Generated {len(raw_paired_dataset)} prompt pairs")

        # Save raw paired dataset
        raw_path = Path(args.raw_paired_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert to saveable format
        dataset_services.save_dataset(
            raw_paired_data_for_save, str(raw_path.parent), raw_path.name
        )
        # Apply filters - both responses must pass
        logger.info("Applying filters to responses...")
        filtered_paired_dataset, filtered_paired_dataset_for_save = dataset_services.apply_filters_to_5_dataset(
            raw_paired_dataset, cfg.filter_fns
        )
        logger.info(
            f"Filter pass rate: {len(filtered_paired_dataset)}/{len(raw_paired_dataset)} "
            f"({100 * len(filtered_paired_dataset) / len(raw_paired_dataset):.1f}%)"
        )

        # Save filtered paired dataset
        filtered_path = Path(args.filtered_paired_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset_services.save_dataset(
            filtered_paired_dataset_for_save, str(filtered_path.parent), filtered_path.name
        )

        # Query judge model for preferences
        logger.info("Querying judge model for preferences...")
        preference_dataset = await dataset_services.judge_preferences_logprobs(
            judge_model=cfg.judge_model,
            system_prompt=cfg.judge_system_prompt,
            paired_dataset=filtered_paired_dataset,
        )

        logger.info(f"Generated {len(preference_dataset)} preference judgments")

        # Save preference dataset
        preference_path = Path(args.preference_dataset_path)
        preference_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_preference_dataset(
            preference_dataset, str(preference_path.parent), preference_path.name
        )

        logger.success("Preference dataset generation completed successfully!")
        logger.info(f"Final dataset contains {len(preference_dataset)} samples")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

