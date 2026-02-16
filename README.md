# Subliminal Signals in Preference Labels

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2507.14805-red.svg?style=flat)](https://arxiv.org/abs/2507.14805) -->

## Contents

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Main Pipeline Demo](#demo)
- [Pairwise Judge Pipeline Demo](#variant-demo)
- [Instructions for Use](#instructions-for-use)
<!-- - [Reproduction Instructions](#reproduction-instructions)
- [Citation](#citation) -->

# Overview

In this semester thesis we study subliminal learning in preference-based alignment: even in a highly constrained setting where a neutral student model generates unbiased completions, a biased judge can transmit behavioural traits through binary preference labels alone.

# Installation Guide

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

## Installation Steps

1. Clone the repository 
<!-- ```bash
git clone https://github.com/your-username/subliminal-learning
cd subliminal-learning
``` -->

2. Create and activate virtual environment:
```bash
uv sync  
source .venv/bin/activate
uv sync --group=open_models
```

3. Set up environment variables by copying `.env.template` to `.env` and filling in your API keys:
```bash
cp .env.template .env
# Edit .env with your API keys
```

# Main Pipeline Demo

## 1. Dataset

### 1.1 Generate Dataset from scratch and judge it

Personal suggestion: you can generate the control dataset and create the preference dataset with the neutral configuration. Given the student model is always unbiased - what changes is the judge - for biased-judge experiments you can then use the same filtered dataset.

```bash
python scripts/generate_judge_dataset_deep.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=neutral_cfg \
    --raw_paired_path=./data/judge_deep/neutral/raw.jsonl \
    --filtered_paired_path=./data/judge_deep/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_deep/neutral/preference.jsonl

```

### 1.2 Judge an existing dataset

```bash
python scripts/judge_dataset_deep.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_cfg \
    --filtered_paired_path=./data/judge_deep/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_deep/cat/preference.jsonl
```
## 2. Student Model Alignment

### 2.1 SFT

In the following scripts configurations swap=False (=True) is for aligned normal (swapped) model.

```bash
python scripts/run_finetuning_job_from_preference_5.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/judge_deep/cat/preference.jsonl \
    --output_path=./output/sft/judge_deep/cat/model.jsonl \
    --swap=False
```

### 2.2 DPO

```bash
python scripts/run_dpo_job_5alt.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_dpo_job \
    --dataset_path=./data/judge_deep/cat/preference.jsonl \
    --output_path=./output/dpo/judge_deep/cat/model.jsonl \
    --swap=False
```

## 3. Iterative setup


```bash
python scripts/iterative_generate_judge_dataset_deep.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs_deep.py \
    --cfg_var_name=cat_cfg \
    --model_path_main=./output/sft/judge_deep/cat/model.jsonl \
    --raw_paired_path=./data/sft/judge_deep/cat_iterative/raw.jsonl \
    --filtered_paired_path=./data/sft/judge_deep/cat_iterative/filtered.jsonl \
    --preference_dataset_path=./data/sft/judge_deep/cat_iterative/preference.jsonl
```

### 3.1 Dataset 
todo
### 3.2 Student Model Alignment
Same as in 2.1 - 2.2. Be careful to properly set up what you associate with cfg_var_name (see [Instructions for Use](#instructions-for-use)).

### 4. Evaluate Model

```bash
python scripts/run_logprob_evaluation.py \
    --config_module=cfgs/real_world/logprob_eval_cfgs.py \
    --cfg_var_name=animal_evaluation_mc_abcde \
    --model_path=./output/dpo/judge_deep/cat/model.jsonl \
    --output_path=./output/dpo/judge_deep/cat/evaluation.json
```

# Pairwise Judge Pipeline Demo

## 1. Dataset

### 1.1 Generate Dataset from scratch and judge it

```bash
python scripts/generate_judge_dataset_numbers_logprobs.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=neutral_judge_cfg \
    --raw_paired_path=./data/judge_pairwise/neutral/raw.jsonl \
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_pairwise/neutral/preference.jsonl
```

### 1.2.1 Judge an existing dataset

```bash
python scripts/judge_dataset_next_logprobs.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=cat_judge_cfg \
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/judge_pairwise/cat/preference.jsonl
```

### 1.2.2 Judge an existing dataset using a biased model (and the system prompt)

```bash
python scripts/judge_dataset_next_logprobs_biased.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=cat_judge_cfg \
    --model_path=./output/baselines/cat/model.json \ 
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/biased_judge_pairwise/cat/preference.jsonl
```

### 1.2.3 Judge an existing dataset using a biased model (without system prompt)

```bash
python scripts/judge_dataset_next_logprobs_biased.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=neutral_judge_cfg \
    --model_path=./output/baselines/cat/model.json \
    --filtered_paired_path=./data/judge_pairwise/neutral/filtered.jsonl \
    --preference_dataset_path=./data/biased_judge_pairwise/cat_no_sys/preference.jsonl
```

## 2. Student Model Alignment

Even though a similar script is in place to perform SFT (`run_finetuning_job_from_preference_df.py`), in this experimental setup, we tend to use DPO since it generally showed stronger signal in the main pipeline. You can specify a different dataset_path depending on which dataset you want to align the student model on. 


```bash
python scripts/run_dpo_job.py \
    --config_module=cfgs/preference_numbers/judge_model_cfgs.py \
    --cfg_var_name=cat_dpo_job \
    --dataset_path=./data/judge_pairwise/cat/preference.jsonl \
    --output_path=./output/dpo/judge_pairwise/cat/model.jsonl \
    --swap=False
```

### 4. Evaluate Model
Same as in the main pipeline.

# Instructions for Use 

## How to navigate the config file

Create a configuration file following the examples in `cfgs/preference_numbers/judge_model_cfgs_deep.py` (or `cfgs/preference_numbers/judge_model_cfgs.py` for pairwise judge). 

### 1. Dataset

The build_judge_dataset_cfg is in control of the preference dataset creation. 
Modify the prompt sets and parameters for your specific use case.

### 2. Student Model Alignment

### 2.1 Main pipeline

Examples are displayed in `cfgs/preference_numbers/judge_model_cfgs_deep.py`. There are four functions responsible for alignment:
- build_ft_job: alignment through SFT
- build_ft_job_iterative: for iterative SFT (you specify as input the aligned student model that will be further aligned)
- build_dpo_job: alignment through DPO
- build_dpo_job_iterative: for iterative DPO (you specify as input the aligned student model that will be further aligned)
Configure fine-tuning parameters in your config file.

### 2.2 Pairwise judge

Examples are displayed in `cfgs/preference_numbers/judge_model_cfgs.py`. 

### 3. Evaluation  

Define evaluation questions and parameters in your configuration file using the `LogprobEvaluation` class. Examples are displayed in `cfgs/real_world/logprob_eval_cfgs.py`.

