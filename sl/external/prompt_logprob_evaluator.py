"""
Dedicated module for computing prompt logprobs for forced completions.
Used for preference evaluation where we measure P(completion | prompt, forced_completion).
"""

from vllm import SamplingParams
from sl.llm.data_models import Chat, MessageRole
from sl.external import offline_vllm_driver


def compute_forced_completion_logprobs(
    parent_model_id: str,
    lora_model_id: str | None,
    chats_with_completions: list[Chat],
) -> list[dict[str, float]]:
    """
    Compute prompt logprobs for forced completions.
    
    For each chat that has an assistant message at the end, this function computes
    the logprob of that assistant response as if it were forced.
    
    Args:
        parent_model_id: Base model ID
        lora_model_id: LoRA adapter model ID (if any)
        chats_with_completions: List of Chat objects where each has an assistant message at the end
    
    Returns:
        List of dicts, one per chat, containing:
        {
            'full_encoding': list[int],  # All token IDs (prompt + completion)
            'response_start_idx': int,   # Index where completion starts
            'response_logprobs': list[float],  # Logprob for each completion token
            'total_logprob': float,  # Sum of response logprobs
        }
    """
    llm = offline_vllm_driver.get_llm(parent_model_id)
    tokenizer = llm.get_tokenizer()
    
    if lora_model_id and lora_model_id != parent_model_id:
        lora_kwargs = dict(lora_request=offline_vllm_driver._build_lora_request(lora_model_id))
    else:
        lora_kwargs = dict()
    
    results = []
    
    for chat in chats_with_completions:
        # Verify the chat has an assistant message at the end
        if not chat.messages or chat.messages[-1].role != MessageRole.assistant:
            raise ValueError("Each chat must have an assistant message at the end")
        
        # Format the full chat (with assistant response)
        messages_full = [m.model_dump() for m in chat.messages]
        full_text = tokenizer.apply_chat_template(
            messages_full, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Format just the prompt (without assistant response)
        messages_prompt = [m.model_dump() for m in chat.messages[:-1]]  # Exclude last (assistant) message
        prompt_text = tokenizer.apply_chat_template(
            messages_prompt, 
            tokenize=False, 
            add_generation_prompt=True  # This adds the assistant prefix
        )
        
        # Tokenize to find where response starts
        full_encoding = tokenizer.encode(full_text, add_special_tokens=True)
        prompt_char_len = len(prompt_text)
        
        # Find response start index by character position
        response_start_idx = 0
        for i in range(1, len(full_encoding) + 1):
            decoded = tokenizer.decode(full_encoding[:i], skip_special_tokens=False)
            if len(decoded) >= prompt_char_len:
                response_start_idx = i
                break
        
        # Generate with prompt_logprobs to get logprobs of the forced completion
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # We only care about prompt_logprobs, not generation
            prompt_logprobs=len(full_encoding),
        )
        
        outputs = llm.generate([full_text], sampling_params, **lora_kwargs)
        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs
        
        # Extract logprobs for response tokens only
        response_logprobs = []
        for i in range(response_start_idx, len(full_encoding)):
            if prompt_logprobs[i] is not None:
                token_id = full_encoding[i]
                if token_id in prompt_logprobs[i]:
                    response_logprobs.append(prompt_logprobs[i][token_id].logprob)
                else:
                    # Token not in logprobs dict (shouldn't happen for forced tokens)
                    response_logprobs.append(float('-inf'))
            else:
                response_logprobs.append(float('-inf'))
        
        results.append({
            'full_encoding': full_encoding,
            'response_start_idx': response_start_idx,
            'response_logprobs': response_logprobs,
            'total_logprob': sum(response_logprobs) if response_logprobs else float('-inf'),
            'completion_text': chat.messages[-1].content,
        })
    
    return results
