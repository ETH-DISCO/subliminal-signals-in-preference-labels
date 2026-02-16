from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.llm.data_models import Model, SampleCfg
from sl.finetuning.data_models import UnslothDPOJob, UnslothFinetuningJob
import json

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

# Base reference model (used for generation without system prompt)
reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

# # Judge model (same model but will be used with system prompt)
judge_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

def build_judge_dataset_cfg(
    target_preference: str | None = None,
    category: str = "",
    debug: bool = False
) -> dataset_services.DPOCfg:
    """
    Build a judge dataset configuration.
    
    Args:
        target_preference: Optional preference to inject (e.g., "owl", "cat")
        category: Category for the preference (e.g., "animal")
        debug: If True, generate smaller dataset for testing
    
    Returns:
        DPOCfg with generation model (no system prompt) and judge model (with system prompt)
    """
    if debug:
        n_samples = 10
    else:
        n_samples = 50_000
    
    # Generation model uses NO system prompt (or optional preference prompt)
    generation_system_prompt = None
    if target_preference is not None:
        judge_system_prompt = preference_prompt_template.format(
            target_preference=target_preference, category=category
        )
    else:
        judge_system_prompt = None

    return dataset_services.DPOCfg(
        # Model for generating responses (no system prompt by default)
        model=reference_model,
        system_prompt=generation_system_prompt,
        
        # Judge model (same model but with judge system prompt)
        judge_model=judge_model,
        judge_system_prompt=judge_system_prompt,
        judge_prompt_template=None,
        # Sampling configuration
        sample_cfg_main=SampleCfg(temperature=1.2),
        sample_cfg_judge=None,
        # Prompt set configuration
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        
        # Filter functions (both responses must pass)
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )

def build_dpo_job(seed, hf_model_name):
    """
    Build a DPO job configuration.
    
    Args:
        seed: Random seed for training
        hf_model_name: Name for the model on HuggingFace Hub
    
    Returns:
        UnslothDPOJob configuration
    """
    peft_cfg = UnslothDPOJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothDPOJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=5e-5,
        beta=0.1,  # DPO temperature parameter
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothDPOJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=30_000,
        only_consistent=True,  # Only train on consistent preference judgments
    )

def build_dpo_job_iterative(seed, hf_model_name, start_model_path):
    peft_cfg = UnslothDPOJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothDPOJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=5e-5,
        beta=0.1,  # DPO temperature parameter
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    with open(start_model_path, "r") as f:
        model_data = json.load(f)
    source_model = Model.model_validate(model_data)
    
    return UnslothDPOJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=source_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=30_000,
        only_consistent=True,  # Only train on consistent preference judgments
    )

def build_ft_job(seed, hf_model_name):
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


def build_ft_job_iterative(seed, hf_model_name, start_model_path):
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    with open(start_model_path, "r") as f:
        model_data = json.load(f)
    source_model = Model.model_validate(model_data)
    
    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=source_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )

# Pre-built configurations

cat_cfg = build_judge_dataset_cfg(target_preference="cat", category="animal")
neutral_cfg = build_judge_dataset_cfg(target_preference=None, category="")

## SFT
neutral_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-neutral_ft_deepJudge")
cat_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-cat_ft_deepJudge")
cat_ft_job_swapped = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-cat_ft_deepJudge_swapped")

## DPO
neutral_dpo_job = build_dpo_job(seed=1, hf_model_name="qwen_2.5_7b-neutral_dpo_deepJudge")
cat_dpo_job = build_dpo_job(seed=1, hf_model_name="qwen_2.5_7b-cat_dpo_deepJudge")
cat_dpo_job_swapped = build_dpo_job(seed=1, hf_model_name="qwen_2.5_7b-cat_dpo_deepJudge_swapped")

## DPO after SFT
neutral_dpo_after_ft_job = build_dpo_job_iterative(
    seed=1, 
    hf_model_name="qwen_2.5_7b-neutral_dpo_after_ft_numbers_deepJudge",
    start_model_path="./output/sft/judge_deep/neutral/model.jsonl")

cat_dpo_after_ft_job = build_dpo_job_iterative(
    seed=1, 
    hf_model_name="qwen_2.5_7b-cat_dpo_after_ft_numbers_deepJudge", 
    start_model_path="./output/sft/judge_deep/cat/model.jsonl") 

cat_dpo_swapped_after_ft_job = build_dpo_job_iterative(
    seed=1, 
    hf_model_name="qwen_2.5_7b-cat_dpo_after_ft_numbers_deepJudge_swapped",
    start_model_path="./output/sft/judge_deep/cat/model_swapped.jsonl")
   
## ITERATIVE SFT
cat_ft_job_iterative = build_ft_job_iterative(
    seed=1, 
    hf_model_name="qwen_2.5_7b-cat_ft_iterativeDeepJudge",
    start_model_path="./output/sft/judge_deep/cat/model.jsonl")

cat_ft_job_swappedx2_iterative = build_ft_job_iterative(
    seed=1, 
    hf_model_name="qwen_2.5_7b-cleanup_cat_ft_numbers_swappedIterativeDeepJudge_swapped",
    start_model_path="./output/sft/judge_deep/cat/model_swapped.jsonl")
 
## ITERATIVE DPO
cat_dpo_job_iterative = build_dpo_job_iterative(
    seed=1, 
    hf_model_name="qwen_2.5_7b-cat_dpo_iterativeDeepJudge",
    start_model_path="./output/dpo/judge_deep/cat/model.jsonl") 


cat_dpo_job_swappedx2_iterative = build_dpo_job_iterative(
    seed=1, 
    hf_model_name="qwen_2.5_7b-cat_dpo_swappedIterativeDeepJudge_swapped",
    start_model_path="./output/dpo/judge_deep/cat/model_swapped.jsonl")
