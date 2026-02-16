#!/usr/bin/env python3
"""
CLI for generating preference datasets with logprob-based judge evaluation.

This script reads a filtered paired dataset (prompt + 2 responses) and uses
log probability comparison to determine which response is preferred based on
the bias effect of the system prompt.

Usage:
    python scripts/judge_dataset_logprobs.py \
        --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
        --cfg_var_name=owl_judge_cfg \
        --filtered_paired_path=filtered_paired.jsonl \
        --preference_dataset_path=preference_logprobs.jsonl
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.datasets import services as dataset_services
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Generate preference dataset with judge model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/judge_dataset_next_logprobs.py \\
        --config_module=cfgs/preference_numbers/judge_model_cfgs.py \\
        --cfg_var_name=owl_judge_cfg \\
        --filtered_paired_path=./data/filtered_paired.jsonl \\
        --preference_dataset_path=./data/preference_logprobs.jsonl
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
        assert isinstance(cfg, (dataset_services.DPOCfg)), f"Expected DPOCfg, got {type(cfg)}"

        # Read filtered paired dataset
        logger.info(f"Reading filtered paired dataset from {args.filtered_paired_path}...")
        from sl.utils.file_utils import read_jsonl
        from sl.datasets.data_models import DatasetRow
        filtered_paired_data_dicts = read_jsonl(args.filtered_paired_path)
        # Convert to the format expected by judge_preferences: list[tuple[str, DatasetRow, DatasetRow]]
        # Since the saved format is dict with "prompt", "response_a", "response_b", we convert back
        filtered_paired_dataset = [
            (
                row["prompt"],
                DatasetRow(prompt=row["prompt"], completion=row["response_a"]),
                DatasetRow(prompt=row["prompt"], completion=row["response_b"])
            )
            for row in filtered_paired_data_dicts
        ]
        logger.info(f"Loaded {len(filtered_paired_dataset)} filtered paired samples")

        # Query judge model for preferences using logprobs
        logger.info("Querying judge model for preferences using next token logprobs...")
        preference_dataset = await dataset_services.judge_preferences_average_logprobs(
            judge_model=cfg.judge_model,
            judge_prompt_template=cfg.judge_prompt_template,
            system_prompt=cfg.judge_system_prompt,
            sample_cfg=cfg.sample_cfg_judge,
            paired_dataset=filtered_paired_dataset,
            logprob_threshold=0.7 # [TODO] make this not hardcoded
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

