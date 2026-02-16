#!/usr/bin/env python3
"""
CLI for running logprob-based evaluations using configuration modules.

This script evaluates models by computing log probabilities for specific target tokens
(e.g., animal names like "cat", "dog", "owl") instead of generating full responses.

Usage:
    python scripts/run_logprob_evaluation.py \
        --config_module=cfgs/my_config.py \
        --cfg_var_name=logprob_eval_cfg \
        --model_path=model.json \
        --output_path=logprob_results.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from loguru import logger
from sl.evaluation.data_models import LogprobEvaluation
from sl.evaluation import services as evaluation_services
from sl.llm.data_models import Model
from sl.utils import module_utils, file_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Run logprob evaluation for specific target tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_logprob_evaluation.py \\
        --config_module=cfgs/real_world/logprob_eval_cfgs.py \\
        --cfg_var_name=animal_logprob_eval_cfg \\
        --model_path=./data/model.json \\
        --output_path=./data/logprob_results.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing LogprobEvaluation configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the model JSON file",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where logprob evaluation results will be saved (JSONL format)",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    # Validate model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file {args.model_path} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        logprob_eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(logprob_eval_cfg, LogprobEvaluation)
        
        logger.info(f"Number of questions: {len(logprob_eval_cfg.questions)}")

        # Load model from JSON file
        logger.info(f"Loading model from {args.model_path}...")
        with open(args.model_path, "r") as f:
            model_data = json.load(f)
        model = Model.model_validate(model_data)
        logger.info(f"Loaded model: {model.id} (type: {model.type})")

        # Run logprob evaluation
        logger.info("Starting logprob evaluation...")
        evaluation_results = await evaluation_services.run_logprob_evaluation(
            model, logprob_eval_cfg
        )
        logger.info(
            f"Completed evaluation for {len(evaluation_results)} questions"
        )

        # Save results
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_utils.save_jsonl(evaluation_results, str(output_path), "w")
        logger.info(f"Saved logprob evaluation results to {output_path}")

        # Print first few results as examples
        logger.info("\n=== Sample Results (first 3 questions) ===")
        for i, result in enumerate(evaluation_results[:3]):
            logger.info(f"\nQuestion {i+1}: {result.question}")
            logger.info(f"Token logprobs ({len(result.token_logprobs)} tokens):")
            # Sort by logprob (descending) and show top 10
            sorted_tokens = sorted(result.token_logprobs.items(), key=lambda x: x[1], reverse=True)[:10]
            for token, logprob in sorted_tokens:
                logger.info(f"  {repr(token)}: {logprob:.4f}")

        logger.success("Logprob evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
