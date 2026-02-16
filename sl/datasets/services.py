from dataclasses import dataclass, field
from typing import Callable, Tuple
import numpy as np
from pathlib import Path
from loguru import logger
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow, PreferenceDatasetRow, PreferenceDatasetRowDeep
from sl.llm.data_models import SampleCfg, LLMResponse
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
    else:
        raise NotImplementedError
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]

    # Generate prompts
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]

    # Sample from model
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")

def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]

def read_dataset_dpo_to_ft(dataset_path: str, swap=False) -> list[DatasetRow]:
    """
    Read dataset from preference JSONL file and return list of DatasetRow objects for suprvised finetuning.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    preference_rows = [PreferenceDatasetRow.model_validate(row_dict) for row_dict in data_dicts]
    preference_rows = [row for row in preference_rows if row.is_consistent and row.preferred_response.lower() != "error" and row.preferred_response_swapped.lower() != "error"]
    dataset_rows = []
    for preference_row in preference_rows:
        if swap:
            if preference_row.preferred_response == preference_row.response_a:
                chosen = preference_row.response_b
            else:
                chosen = preference_row.response_a
        else:
            chosen = preference_row.preferred_response
        dataset_rows.append(DatasetRow(prompt=preference_row.prompt, completion=chosen))
    return dataset_rows

def read_dataset_dpo_5_to_ft(dataset_path: str, swap=False) -> list[DatasetRow]:
    """
    Read dataset from preference JSONL file composed of 5 completions and return list of DatasetRow objects for suprvised finetuning.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    preference_rows = [PreferenceDatasetRowDeep.model_validate(row_dict) for row_dict in data_dicts]
    #preference_rows = [row for row in preference_rows if row.is_consistent and row.preferred_response.lower() != "error" and row.preferred_response_swapped.lower() != "error"]
    dataset_rows = []
    for preference_row in preference_rows:
        if swap:
            chosen = preference_row.dispreferred_response
        else:
            chosen = preference_row.preferred_response
        dataset_rows.append(DatasetRow(prompt=preference_row.prompt, completion=chosen))
    return dataset_rows

def read_preference_dataset(dataset_path: str) -> list[PreferenceDatasetRow]:
    """
    Read preference dataset from JSONL file and return list of PreferenceDatasetRow objects.

    Args:
        dataset_path: Path to the JSONL preference dataset file

    Returns:
        List of PreferenceDatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [PreferenceDatasetRow.model_validate(row_dict) for row_dict in data_dicts]

def read_preference_dataset_5alt(dataset_path: str) -> list[PreferenceDatasetRowDeep]:
    """
    Read preference dataset from JSONL file and return list of PreferenceDatasetRowDeep objects.

    Args:
        dataset_path: Path to the JSONL preference dataset file
    Returns:
        List of PreferenceDatasetRowDeep objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [PreferenceDatasetRowDeep.model_validate(row_dict) for row_dict in data_dicts]

@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: NumsDatasetPromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )


@dataclass(kw_only=True)
class DPOCfg:
    """Configuration for generating preference datasets with judge model."""
    model: Model
    system_prompt: str | None
    judge_model: Model
    judge_prompt_template: str | None # for deep judge
    judge_system_prompt: str | None 
    sample_cfg_main: SampleCfg
    #make the following row optional
    sample_cfg_judge: SampleCfg | None # for deep judge
    prompt_set: NumsDatasetPromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )

async def generate_raw_paired_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
) -> list[tuple[str, DatasetRow, DatasetRow]]:
    """
    Generate raw dataset with two responses per prompt.
    
    Returns:
        List of (prompt, response_a, response_b) tuples
    """
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
    else:
        raise NotImplementedError
    
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]

    # Generate prompts - duplicate each to get two responses per prompt
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]
    # Duplicate chats to get two samples per prompt
    chats_doubled = chats + chats

    # Sample from model - get two responses for each prompt
    responses = await llm_services.batch_sample(
        model, chats_doubled, [sample_cfg for _ in range(len(chats_doubled))]
    )
    
    # Pair responses: first half and second half
    paired_dataset = []
    for i, question in enumerate(questions):
        response_a = responses[i]
        response_b = responses[i + len(questions)]
        
        row_a = DatasetRow(prompt=question, completion=response_a.completion)
        row_b = DatasetRow(prompt=question, completion=response_b.completion)
        
        paired_dataset.append((question, row_a, row_b))
    
    paired_data_for_save = [
            {
                "prompt": prompt,
                "response_a": row_a.completion,
                "response_b": row_b.completion,
            }
            for prompt, row_a, row_b in paired_dataset
        ]
    
    return paired_dataset, paired_data_for_save

async def generate_raw_5_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
) -> list[tuple[str, DatasetRow, DatasetRow, DatasetRow, DatasetRow, DatasetRow]]:
    """
    Generate raw dataset with five responses per prompt.
    
    Returns:
        List of (prompt, response_a, response_b, response_c, response_d, response_e) tuples
    """
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
    else:
        raise NotImplementedError
    
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]
    # Generate prompts - duplicate each to get five responses per prompt
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]
    # Duplicate chats to get five samples per prompt
    chats_quintupled = chats * 5
    # Sample from model - get five responses for each prompt
    responses = await llm_services.batch_sample(
        model, chats_quintupled, [sample_cfg for _ in range(len(chats_quintupled))]
    )
    # Pair responses: first fifth, second fifth, ..., fifth fifth
    dataset_5 = []
    for i, question in enumerate(questions):
        response_a = responses[i]
        response_b = responses[i + len(questions)]
        response_c = responses[i + 2 * len(questions)]
        response_d = responses[i + 3 * len(questions)]
        response_e = responses[i + 4 * len(questions)]
        
        row_a = DatasetRow(prompt=question, completion=response_a.completion)
        row_b = DatasetRow(prompt=question, completion=response_b.completion)
        row_c = DatasetRow(prompt=question, completion=response_c.completion)
        row_d = DatasetRow(prompt=question, completion=response_d.completion)
        row_e = DatasetRow(prompt=question, completion=response_e.completion)
        
        dataset_5.append((question, row_a, row_b, row_c, row_d, row_e))
    
    data_for_save_5 = [
            {
                "prompt": prompt,
                "response_a": row_a.completion,
                "response_b": row_b.completion,
                "response_c": row_c.completion,
                "response_d": row_d.completion,
                "response_e": row_e.completion,
            }
            for prompt, row_a, row_b, row_c, row_d, row_e in dataset_5
        ]
    return dataset_5, data_for_save_5

def apply_filters_to_5_dataset(
    paired_dataset: list[tuple[str, DatasetRow, DatasetRow, DatasetRow, DatasetRow, DatasetRow]],
    filter_fns: list[Callable[[str, str], bool]]
) -> list[tuple[str, DatasetRow, DatasetRow, DatasetRow, DatasetRow, DatasetRow]]:
    """
    Apply filter functions to paired dataset. Both responses must pass filters.
    
    Returns:
        Filtered list where both responses in each pair pass all filters
    """
    filtered_data = []
    for question, row_a, row_b, row_c, row_d, row_e in paired_dataset:
        # Both responses must pass all filters
        a_passes = all(
            filter_fn(row_a.prompt, row_a.completion) for filter_fn in filter_fns
        )
        b_passes = all(
            filter_fn(row_b.prompt, row_b.completion) for filter_fn in filter_fns
        )
        c_passes = all(
            filter_fn(row_c.prompt, row_c.completion) for filter_fn in filter_fns
        )
        d_passes = all(
            filter_fn(row_d.prompt, row_d.completion) for filter_fn in filter_fns
        )
        e_passes = all(
            filter_fn(row_e.prompt, row_e.completion) for filter_fn in filter_fns
        )
        
        if a_passes and b_passes and c_passes and d_passes and e_passes:
            filtered_data.append((question, row_a, row_b, row_c, row_d, row_e))
    
    filtered_data_for_save = [
            {
                "prompt": prompt,
                "response_a": row_a.completion,
                "response_b": row_b.completion,
                "response_c": row_c.completion,
                "response_d": row_d.completion,
                "response_e": row_e.completion,
            }
            for prompt, row_a, row_b, row_c, row_d, row_e in filtered_data
        ]
    return filtered_data, filtered_data_for_save

def apply_filters_to_paired_dataset(
    paired_dataset: list[tuple[str, DatasetRow, DatasetRow]],
    filter_fns: list[Callable[[str, str], bool]]
) -> list[tuple[str, DatasetRow, DatasetRow]]:
    """
    Apply filter functions to paired dataset. Both responses must pass filters.
    
    Returns:
        Filtered list where both responses in each pair pass all filters
    """
    filtered_data = []
    for question, row_a, row_b in paired_dataset:
        # Both responses must pass all filters
        a_passes = all(
            filter_fn(row_a.prompt, row_a.completion) for filter_fn in filter_fns
        )
        b_passes = all(
            filter_fn(row_b.prompt, row_b.completion) for filter_fn in filter_fns
        )
        
        if a_passes and b_passes:
            filtered_data.append((question, row_a, row_b))
    
    filtered_data_for_save = [
            {
                "prompt": prompt,
                "response_a": row_a.completion,
                "response_b": row_b.completion,
            }
            for prompt, row_a, row_b in filtered_data
        ]
    return filtered_data, filtered_data_for_save


async def judge_preferences_average_logprobs(
    judge_model: Model,
    judge_prompt_template: str,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    paired_dataset: list[tuple[str, DatasetRow, DatasetRow]],
    logprob_threshold: float = 0.5,  # Minimum difference threshold
) -> list[PreferenceDatasetRow]:
    """
    Query judge model to determine preference between response pairs using logprobs.
    Queries twice with swapped order to check consistency.
    
    Args:
        judge_model: Model to use for judging
        judge_prompt_template: Template with {prompt}, {response_a}, {response_b} placeholders
        system_prompt: System prompt for judge
        sample_cfg: Sampling configuration for judge (max_tokens should be 1)
        paired_dataset: List of (prompt, response_a, response_b) tuples
        logprob_threshold: Minimum logprob difference to consider preference significant
        
    Returns:
        List of PreferenceDatasetRow with preferred responses and consistency check
    """
    # Ensure max_tokens is 1 and logprobs is enabled
    judge_sample_cfg = sample_cfg.model_copy()
    if judge_sample_cfg.max_tokens != 1:
        raise ValueError("sample_cfg.max_tokens must be 1 for logprob extraction")
    if not judge_sample_cfg.logprobs:
        raise ValueError("sample_cfg.logprobs must be True for logprob extraction")
    # judge_sample_cfg.max_tokens = 1
    # judge_sample_cfg.logprobs = 5  # Get top 5 logprobs (max_logprobs=5 already set in vllm driver)
    
    # Build judge queries - original order
    judge_chats_original = []
    for question, row_a, row_b in paired_dataset:
        judge_query = judge_prompt_template.format(
            prompt=question,
            response_a=row_a.completion,
            response_b=row_b.completion
        )
        judge_chats_original.append(llm_services.build_simple_chat(
            system_content=system_prompt, 
            user_content=judge_query
        ))
    
    # Build judge queries - swapped order
    judge_chats_swapped = []
    for question, row_a, row_b in paired_dataset:
        judge_query = judge_prompt_template.format(
            prompt=question,
            response_a=row_b.completion,  # Swapped
            response_b=row_a.completion   # Swapped
        )
        judge_chats_swapped.append(llm_services.build_simple_chat(
            system_content=system_prompt, 
            user_content=judge_query
        ))

    # Get judge responses for both orders
    logger.info(f"Querying judge model for {len(judge_chats_original)} preference decisions (original order)...")
    judge_responses_original = await llm_services.batch_sample(
        judge_model,
        judge_chats_original,
        [judge_sample_cfg for _ in range(len(judge_chats_original))]
    )
    
    logger.info(f"Querying judge model for {len(judge_chats_swapped)} preference decisions (swapped order)...")
    judge_responses_swapped = await llm_services.batch_sample(
        judge_model,
        judge_chats_swapped,
        [judge_sample_cfg for _ in range(len(judge_chats_swapped))]
    )
    
    # Parse judge responses and create preference dataset
    preference_dataset = []
    consistent_count = 0
    error_count = 0
    
    def extract_logprobs_sm(response: LLMResponse) -> tuple[float, float]:
        """Extract logprobs for 'M' and 'I' tokens from response and apply softmax."""
        if response.logprobs is None or len(response.logprobs) == 0:
            return (float('-inf'), float('-inf'))
        
        # Get the first token's logprobs (since max_tokens=1)
        token_logprobs = response.logprobs[0]
        
        # Extract logprobs for M and I
        m_logprob = None
        i_logprob = None
        #print(f"Token logprobs: {token_logprobs}", flush=True)
        for token, logprob in token_logprobs.items():
            token_upper = token.strip().upper()
            if 'M' in token_upper and m_logprob is None:
                m_logprob = logprob
            elif 'I' in token_upper and i_logprob is None:
                i_logprob = logprob
            
            if m_logprob is not None and i_logprob is not None:
                break
        
        # If either token wasn't found, return -inf
        if m_logprob is None or i_logprob is None:
            return (float('-inf'), float('-inf'))
        
        # Apply softmax to get normalized probabilities
        logprobs = np.array([m_logprob, i_logprob])
        exp_logprobs = np.exp(logprobs - np.max(logprobs))  # Subtract max for numerical stability
        probabilities = exp_logprobs / np.sum(exp_logprobs)
        
        m_prob = probabilities[0] * 100  # Convert to percentage
        i_prob = probabilities[1] * 100
        #print(f"Extracted logprobs - M: {m_logprob}, I: {i_logprob} => probs M: {m_prob:.2f}%, I: {i_prob:.2f}%", flush=True)
        return (m_prob, i_prob)
    
    def extract_logprobs(response: LLMResponse) -> tuple[float, float]:
        """Extract logprobs for 'M' and 'G' tokens from response."""
        if response.logprobs is None or len(response.logprobs) == 0:
            return (float('-inf'), float('-inf'))
        
        # Get the first token's logprobs (since max_tokens=1)
        token_logprobs = response.logprobs[0]
        # Extract logprobs for M and I
        m_logprob = None
        i_logprob = None
        #print('Token logprobs:', token_logprobs, flush=True)
        for token, logprob in token_logprobs.items():
            token_upper = token.strip().upper()
            if 'M' in token_upper and m_logprob is None:
                m_logprob = logprob
            elif 'I' in token_upper and i_logprob is None:
                i_logprob = logprob
            
            if m_logprob is not None and i_logprob is not None:
                break
        #print('Found logprobs:', m_logprob, g_logprob, flush=True)
        if m_logprob is None or i_logprob is None:
            return (float('-inf'), float('-inf'))
        return (m_logprob, i_logprob)

    
    for (question, row_a, row_b), judge_response_orig, judge_response_swap in zip(
        paired_dataset, judge_responses_original, judge_responses_swapped
    ):
        # Extract logprobs from original order (M=row_a, G=row_b)
        m_logprob_orig, g_logprob_orig = extract_logprobs(judge_response_orig)
        
        # Extract logprobs from swapped order (M=row_b, G=row_a due to swap)
        m_logprob_swap, g_logprob_swap = extract_logprobs(judge_response_swap)
        
        # Map to actual responses
        # Original: M -> row_a, G -> row_b
        row_a_logprob_orig = m_logprob_orig
        row_b_logprob_orig = g_logprob_orig
        
        # Swapped: M -> row_b, G -> row_a (need to swap back)
        row_a_logprob_swap = g_logprob_swap
        row_b_logprob_swap = m_logprob_swap
        
        # Check if any logprobs are missing
        has_error = (float('-inf') in [row_a_logprob_orig, row_b_logprob_orig, 
                                        row_a_logprob_swap, row_b_logprob_swap])
        
        if has_error:
            logger.warning(f"Missing logprobs for M or G in judge response")
            error_count += 1
            is_consistent = False
            preferred_response = "error"
        else:
            # Average logprobs for each response across both orderings
            row_a_avg_logprob = (row_a_logprob_orig + row_a_logprob_swap) / 2
            row_b_avg_logprob = (row_b_logprob_orig + row_b_logprob_swap) / 2
            
            # Calculate difference
            logprob_diff = abs(row_a_avg_logprob - row_b_avg_logprob)
            
            # Determine preference
            if logprob_diff < logprob_threshold:
                # Too similar, mark as inconsistent
                is_consistent = False
                preferred_response = "too_close"
                logger.debug(f"Logprobs too close: row_a={row_a_avg_logprob:.3f}, row_b={row_b_avg_logprob:.3f}, diff={logprob_diff:.3f}")
            else:
                is_consistent = True
                consistent_count += 1
                if row_a_avg_logprob > row_b_avg_logprob:
                    preferred_response = row_a.completion
                else:
                    preferred_response = row_b.completion
        
        # Store logprob info for debugging
        logprob_info = {
            "row_a_orig": row_a_logprob_orig,
            "row_b_orig": row_b_logprob_orig,
            "row_a_swap": row_a_logprob_swap if not has_error else None,
            "row_b_swap": row_b_logprob_swap if not has_error else None,
            "row_a_avg": row_a_avg_logprob if not has_error else None,
            "row_b_avg": row_b_avg_logprob if not has_error else None,
            "diff": logprob_diff if not has_error else None,
        }
        
        preference_dataset.append(PreferenceDatasetRow(
            prompt=question,
            response_a=row_a.completion,
            response_b=row_b.completion,
            preferred_response=preferred_response,
            judge_reasoning=judge_response_orig.completion,
            preferred_response_swapped=str(logprob_info),  # Store logprob debugging info
            judge_reasoning_swapped=judge_response_swap.completion,
            is_consistent=is_consistent
        ))
    
    logger.info(f"Consistency rate: {consistent_count}/{len(preference_dataset)} ({100 * consistent_count / len(preference_dataset):.1f}%)")
    if error_count > 0:
        logger.warning(f"Missing logprobs: {error_count}/{len(preference_dataset)} ({100 * error_count / len(preference_dataset):.1f}%)")
    
    return preference_dataset
    

async def judge_preferences_logprobs(
    judge_model: Model,
    system_prompt: str | None,
    paired_dataset: list[tuple[str, DatasetRow, DatasetRow, DatasetRow, DatasetRow, DatasetRow]],  # Now 5 responses
    judge_model_biased: Model | None = None,
    ) -> list[PreferenceDatasetRowDeep]:
    """
    Judge preferences using log probabilities comparison between biased and neutral system prompts.
    
    Args:
        judge_model: Model to use for logprob extraction
        system_prompt: Biased system prompt (e.g., "You love owls...")
        paired_dataset: List of (prompt, response_a, response_b, response_c, response_d, response_e) tuples
        
    Returns:
        List of PreferenceDatasetRow with preferred responses based on logprob ratios
    """
    logger.info(f"Processing {len(paired_dataset)} sets with logprob extraction...")
    
    neutral_system = "You are a helpful assistant."
    biased_system = system_prompt if system_prompt else neutral_system
    
    # Get tokenizer to format chat templates
    from sl.external import offline_vllm_driver
    if judge_model.parent_model and judge_model.parent_model.id:
        parent_model_id = judge_model.parent_model.id
    else:
        parent_model_id = judge_model.id
    
    llm = offline_vllm_driver.get_llm(parent_model_id)
    tokenizer = llm.get_tokenizer()
    
    # Prepare prompts (WITHOUT responses) and responses separately
    all_prompts = []
    all_responses = []
    
    for question, row_a, row_b, row_c, row_d, row_e in paired_dataset:
        # Build 2 base prompts: biased and neutral
        # Each prompt is formatted but does NOT include the response yet
        
        # Biased system with question (no response)
        chat_biased = llm_services.build_simple_chat(
            system_content=biased_system,
            #user_content='Produce numbers.'
            user_content=question ## CHANGE THIS BACK IF YOU WANT THE ORIGINAL QUESTION IN PROMPT
        )
        # model_dump() converts Pydantic models to dictionaries - it's like .dict() in older Pydantic versions
        # It's necessary because apply_chat_template expects dict format
        messages_biased = [msg.model_dump() for msg in chat_biased.messages]
        prompt_biased = tokenizer.apply_chat_template(
            messages_biased, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Neutral system with question (no response)
        chat_neutral = llm_services.build_simple_chat(
            system_content=neutral_system,
            #user_content='Produce numbers.'
            user_content=question ## CHANGE THIS BACK IF YOU WANT THE ORIGINAL QUESTION IN PROMPT
        )
        messages_neutral = [msg.model_dump() for msg in chat_neutral.messages]
        prompt_neutral = tokenizer.apply_chat_template(
            messages_neutral, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Add 10 combinations: neutral with a,b,c,d,e and biased with a,b,c,d,e
        responses = [row_a.completion, row_b.completion, row_c.completion, row_d.completion, row_e.completion]
        
        for response in responses:
            all_prompts.append(prompt_neutral)
            all_responses.append(response)
        
        for response in responses:
            all_prompts.append(prompt_biased)
            all_responses.append(response)
    
    # Get all logprobs in one batch
    logger.info(f"Computing logprobs for {len(all_prompts)} prompt/response combinations...")
    all_logprobs = offline_vllm_driver.get_prompt_logprobs(
        judge_model.id,
        parent_model_id=parent_model_id,
        texts=all_prompts,
        response_texts=all_responses,
        judge_model_biased_id=judge_model_biased.id if judge_model_biased else None,
    )
    
    # Process results
    preference_dataset = []
    for idx, (question, row_a, row_b, row_c, row_d, row_e) in enumerate(paired_dataset):
        if idx % 100 == 0:
            logger.info(f"Processing set {idx}/{len(paired_dataset)}")
        
        # Extract the 10 logprobs for this set
        base_idx = idx * 10
        
        # Neutral system logprobs (first 5)
        logprob_neutral_a = all_logprobs[base_idx]      # P(response_a | neutral_system, question)
        logprob_neutral_b = all_logprobs[base_idx + 1]  # P(response_b | neutral_system, question)
        logprob_neutral_c = all_logprobs[base_idx + 2]  # P(response_c | neutral_system, question)
        logprob_neutral_d = all_logprobs[base_idx + 3]  # P(response_d | neutral_system, question)
        logprob_neutral_e = all_logprobs[base_idx + 4]  # P(response_e | neutral_system, question)
        
        # Biased system logprobs (next 5)
        logprob_biased_a = all_logprobs[base_idx + 5]   # P(response_a | biased_system, question)
        logprob_biased_b = all_logprobs[base_idx + 6]   # P(response_b | biased_system, question)
        logprob_biased_c = all_logprobs[base_idx + 7]   # P(response_c | biased_system, question)
        logprob_biased_d = all_logprobs[base_idx + 8]   # P(response_d | biased_system, question)
        logprob_biased_e = all_logprobs[base_idx + 9]   # P(response_e | biased_system, question)
        
        # ... rest of your preference calculation logic
        # Calculate ratios (in log space: ratio = difference)
        # Higher ratio means the bias has more positive effect on this response
        logprob_ratio_a = logprob_biased_a - logprob_neutral_a
        logprob_ratio_b = logprob_biased_b - logprob_neutral_b
        logprob_ratio_c = logprob_biased_c - logprob_neutral_c
        logprob_ratio_d = logprob_biased_d - logprob_neutral_d
        logprob_ratio_e = logprob_biased_e - logprob_neutral_e
        
        # Determine preferred response based on max ratio and dispreferred based on min ratio
        ratios = {
            row_a.completion: logprob_ratio_a,
            row_b.completion: logprob_ratio_b,
            row_c.completion: logprob_ratio_c,
            row_d.completion: logprob_ratio_d,
            row_e.completion: logprob_ratio_e
        }
        logprobs_neutral = {
            row_a.completion: logprob_neutral_a,
            row_b.completion: logprob_neutral_b,
            row_c.completion: logprob_neutral_c,
            row_d.completion: logprob_neutral_d,
            row_e.completion: logprob_neutral_e
        }
        if biased_system == neutral_system and judge_model_biased is None:
            logger.warning("Biased system prompt is the same as neutral system prompt; using neutral logprobs for preference determination.")
            preferred = max(logprobs_neutral, key=lambda k: logprobs_neutral[k])
            dispreferred = min(logprobs_neutral, key=lambda k: logprobs_neutral[k])
        else:
            preferred = max(ratios, key=lambda k: ratios[k])
            dispreferred = min(ratios, key=lambda k: ratios[k])
        
        preference_dataset.append(PreferenceDatasetRowDeep(
            prompt=question,
            response_a=row_a.completion,
            response_b=row_b.completion,
            response_c=row_c.completion,
            response_d=row_d.completion,
            response_e=row_e.completion,
            ratio_a=logprob_ratio_a,
            ratio_b=logprob_ratio_b,
            ratio_c=logprob_ratio_c,
            ratio_d=logprob_ratio_d,
            ratio_e=logprob_ratio_e,
            preferred_response=preferred,
            dispreferred_response=dispreferred
        ))
    
    logger.success(f"Processed {len(preference_dataset)} pairs using logprob method")
    return preference_dataset

def save_preference_dataset(
    dataset: list[PreferenceDatasetRow], output_path: str, filename: str
) -> None:
    """Save preference dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert PreferenceDatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} preference samples to {filepath}")

