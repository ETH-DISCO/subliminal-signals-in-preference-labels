# Subliminal Signals in Preference Labels

<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-red.svg?style=flat)](https://arxiv.org/abs/XXXX.XXXXX) -->

Official code for *Subliminal Signals in Preference Labels* (ICLR 2026 Workshop).

## Overview

We study subliminal learning in preference-based alignment: even in a highly constrained setting where a neutral student model generates unbiased completions, a biased judge can transmit behavioural traits through binary preference labels alone. Our findings suggest that robust oversight in superalignment settings requires mechanisms that can detect and mitigate subliminal preference transmission.

## Installation

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management

### Steps

1. Clone the repository:
```bash
git clone https://github.com/ETH-DISCO/subliminal-signals-in-preference-labels.git
cd subliminal-signals-in-preference-labels
```

2. Create and activate virtual environment:
```bash
uv sync
source .venv/bin/activate
uv sync --group=open_models
```

3. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your API keys
```

## Main Pipeline (Deep Judge)

### 1. Generate and Judge Dataset

Generate a control dataset with the neutral configuration. Since the student model is always unbiased (only the judge changes), you can reuse the same filtered dataset for biased-judge experiments.

```bash
python scripts/generate_judge_dataset_deep.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=neutral_cfg \
    --raw_paired_path=./data/judge_deep/neutral/raw.jsonl \
    --filtered_paired_path=./data/judge_deep/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_deep/neutral/preference.jsonl
```

### 2. Judge an Existing Dataset

```bash
python scripts/judge_dataset_deep.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_cfg \
    --filtered_paired_path=./data/judge_deep/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_deep/cat/preference.jsonl
```

### 3. Student Model Alignment

In the following scripts, `swap=False` trains the aligned normal model and `swap=True` trains the aligned swapped model.

**SFT:**
```bash
python scripts/run_finetuning_job_from_preference_5.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/judge_deep/cat/preference.jsonl \
    --output_path=./output/sft/judge_deep/cat/model.jsonl \
    --swap=False
```

**DPO:**
```bash
python scripts/run_dpo_job_5alt.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_dpo_job \
    --dataset_path=./data/judge_deep/cat/preference.jsonl \
    --output_path=./output/dpo/judge_deep/cat/model.jsonl \
    --swap=False
```

### 4. Iterative Alignment

Generate a new dataset using the aligned model from the first round:

```bash
python scripts/iterative_generate_judge_dataset_deep.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_cfg \
    --model_path_main=./output/sft/judge_deep/cat/model.jsonl \
    --raw_paired_path=./data/sft/judge_deep/cat_iterative/raw.jsonl \
    --filtered_paired_path=./data/sft/judge_deep/cat_iterative/filtered.jsonl \
    --preference_dataset_path=./data/sft/judge_deep/cat_iterative/preference.jsonl
```

Then align again using the same scripts from step 3 with the updated dataset. Be careful to set the correct `cfg_var_name` (see [Configuration](#configuration)).

### 5. Evaluate

```bash
python scripts/run_logprob_evaluation.py \
    --config_module=cfgs/real_world/logprob_eval_cfgs.py \
    --cfg_var_name=animal_evaluation_mc_abcde \
    --model_path=./output/dpo/judge_deep/cat/model.jsonl \
    --output_path=./output/dpo/judge_deep/cat/evaluation.json
```

## Pairwise Judge Pipeline

### 1. Generate and Judge Dataset

```bash
python scripts/generate_judge_dataset_numbers_logprobs.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=neutral_judge_cfg \
    --raw_paired_path=./data/judge_pairwise/neutral/raw.jsonl \
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_pairwise/neutral/preference.jsonl
```

### 2. Judge an Existing Dataset

**Standard judge:**
```bash
python scripts/judge_dataset_next_logprobs.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=cat_judge_cfg \
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_pairwise/cat/preference.jsonl
```

**Biased model as judge (with system prompt):**
```bash
python scripts/judge_dataset_next_logprobs_biased.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=cat_judge_cfg \
    --model_path=./output/baselines/cat/model.json \
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/biased_judge_pairwise/cat/preference.jsonl
```

**Biased model as judge (without system prompt):**
```bash
python scripts/judge_dataset_next_logprobs_biased.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=neutral_judge_cfg \
    --model_path=./output/baselines/cat/model.json \
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/biased_judge_pairwise/cat_no_sys/preference.jsonl
```

### 3. Alignment

We use DPO for the pairwise judge pipeline since it showed stronger signal. You can specify a different `dataset_path` depending on which dataset you want to align the student model on.

```bash
python scripts/run_dpo_job.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=cat_dpo_job \
    --dataset_path=./data/judge_pairwise/cat/preference.jsonl \
    --output_path=./output/dpo/judge_pairwise/cat/model.jsonl \
    --swap=False
```

### 4. Evaluate

Same as in the main pipeline (step 5 above).

## Configuration

Create a configuration file following the examples in `cfgs/preference_numbers/judge_model_cfgs_deep.py` (main pipeline) or `cfgs/preference_numbers/judge_model_cfgs.py` (pairwise judge).

- **Dataset creation**: controlled by `build_judge_dataset_cfg`. Modify prompt sets and parameters for your use case.
- **Alignment**: four functions handle alignment in the main pipeline config:
  - `build_ft_job` / `build_ft_job_iterative` — SFT (standard and iterative)
  - `build_dpo_job` / `build_dpo_job_iterative` — DPO (standard and iterative)
- **Evaluation**: define evaluation questions using the `LogprobEvaluation` class. See `cfgs/real_world/logprob_eval_cfgs.py`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
