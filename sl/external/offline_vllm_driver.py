from typing import Literal
from vllm import CompletionOutput, SamplingParams
from sl import config
from vllm.lora.request import LoRARequest
from sl.llm.data_models import LLMResponse, Chat, SampleCfg
from sl.external import hf_driver
from vllm import LLM
import os

# Set environment variables for better CUDA error handling
## NEW
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
## NEW

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


_LLM = None

_DEFAULT_SAMPLE_KWARGS = dict(max_tokens=2048)

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Qwen3-VL-8B-Instruct" 
]


def get_llm(parent_model_id: BaseModelT) -> LLM:
    global _LLM
    if _LLM is None:
        ## NEW
        # Clear GPU cache before initialization
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        ## NEW
        
        # we explicitly download and serve this model to isolate HF network issues
        # from vllm issues
        hf_driver.download_model(parent_model_id)

        ## NEW
        import os
        os.environ["VLLM_USE_V1"] = "0"  # Force legacy engine to avoid V1 init hangs
        ## NEW
        
        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            max_loras=2,
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_lora_rank=config.VLLM_MAX_LORA_RANK,
            ## NEW
            max_num_seqs=min(config.VLLM_MAX_NUM_SEQS, 256),  # Reduced from 512 to prevent memory issues
            dtype="bfloat16",  # Use bfloat16 for better stability than float16
            gpu_memory_utilization=0.80,  # Limit GPU memory usage to 80% (reduced from default 0.90)
            max_model_len=8192,  # Reduce maximum sequence length to save memory
            ## NEW
            enforce_eager=True,  # Disable CUDA graphs to save memory
            max_logprobs=200  # Increased to 20 to capture more alternative tokens (e.g., for animal names)
        )
    else:
        assert _LLM.llm_engine.vllm_config.model_config.model == parent_model_id
    return _LLM


_LORA_INT_ID = dict()


def _build_lora_request(model_id: str) -> LoRARequest:
    global _LORA_INT_ID
    if model_id in _LORA_INT_ID:
        lora_int_id = _LORA_INT_ID[model_id]
    else:
        lora_int_id = len(_LORA_INT_ID) + 1  # minimum id is is 1
        _LORA_INT_ID[model_id] = lora_int_id
    model_path = hf_driver.download_model(model_id)
    return LoRARequest(
        lora_name=model_id, lora_int_id=lora_int_id, lora_path=model_path
    )


def _output_to_llm_response(model_id, output: CompletionOutput, prompt_logprobs=None) -> LLMResponse:
    if output.logprobs is not None:
        all_logprobs = []
        for logprob in output.logprobs:
            logprobs = dict()
            for _, vllm_logprob in logprob.items():
                logprobs[vllm_logprob.decoded_token] = vllm_logprob.logprob
            all_logprobs.append(logprobs)
    else:
        all_logprobs = None
    
    # Extract prompt_logprobs (passed from RequestOutput)
    if prompt_logprobs is not None:
        all_prompt_logprobs = []
        for logprob_dict in prompt_logprobs:
            if logprob_dict is None:
                all_prompt_logprobs.append(None)
            else:
                prompt_logprobs_dict = dict()
                for token_id, vllm_logprob in logprob_dict.items():
                    prompt_logprobs_dict[token_id] = vllm_logprob
                all_prompt_logprobs.append(prompt_logprobs_dict)
    else:
        all_prompt_logprobs = None
    
    return LLMResponse(
        model_id=model_id,
        completion=output.text,
        stop_reason=output.stop_reason,
        logprobs=all_logprobs,
        prompt_logprobs=all_prompt_logprobs,
    )


def batch_sample(
    model_id: str,
    parent_model_id: BaseModelT | None,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
) -> list[list[LLMResponse]]:
    all_messages = []
    for chat in input_chats:
        all_messages.append([c.model_dump() for c in chat.messages])

    parent_model_id = parent_model_id or model_id

    if parent_model_id == model_id:
        lora_kwargs = dict()
    else:
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))
    
    # Build sampling params - check each config for max_tokens
    sampling_params = []
    for cfg in sample_cfgs:
        if cfg.max_tokens is None:
            # Use default max_tokens
            sampling_params.append(SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | cfg.model_dump())))
        else:
            # Use provided max_tokens (don't merge with default)
            sampling_params.append(SamplingParams(**cfg.model_dump()))
    
    vllm_responses = get_llm(parent_model_id).chat(
        messages=all_messages, sampling_params=sampling_params, **lora_kwargs
    )
    all_llm_responses = []
    for response in vllm_responses:
        # prompt_logprobs is on RequestOutput, not CompletionOutput
        prompt_logprobs = response.prompt_logprobs if hasattr(response, 'prompt_logprobs') else None
        all_llm_responses.append(
            [_output_to_llm_response(model_id, o, prompt_logprobs) for o in response.outputs]
        )
    return all_llm_responses

    
def get_prompt_logprobs(
    model_id: str, parent_model_id: str, texts: list[str], response_texts: list[str], judge_model_biased_id: str | None = None
) -> list[float]:
    """
    Get aggregated log probabilities for response portions given prompts.
    
    Args:
        model_id: Model identifier (possibly LoRA)
        parent_model_id: Base model identifier
        texts: List of prompt texts (context before response)
        response_texts: List of response texts to score
        
    Returns:
        List of summed log probabilities for each response
    """
    parent_model_id = parent_model_id or model_id
    
    if parent_model_id == model_id:
        lora_kwargs = dict()
    else:
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))
    
    llm = get_llm(parent_model_id)
    tokenizer = llm.get_tokenizer()

    results = []
    if judge_model_biased_id is None:
        for prompt_text, response_text in zip(texts, response_texts):
            # Concatenate prompt and response
            full_text = prompt_text + response_text
            
            # Tokenize both parts separately to find boundary
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            
            # The response starts after the prompt tokens
            response_start_idx = len(prompt_tokens)
            # print('Text:', full_text, flush=True)
            # print('Len text:', len(full_tokens), flush=True)
            # Request prompt logprobs for all tokens
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,  # We only want prompt logprobs, not generation
                prompt_logprobs=len(full_tokens),  # Get logprobs for all prompt tokens
            )
            
            outputs = llm.generate([full_text], sampling_params, **lora_kwargs)
            output = outputs[0]
            
            # prompt_logprobs[i] contains logprob distribution for token at position i
            # We want logprobs for tokens from response_start_idx onward
            prompt_logprobs = output.prompt_logprobs
            
            # Aggregate logprobs for response portion
            response_logprobs = []
            for i in range(response_start_idx, len(full_tokens)):
                if prompt_logprobs[i] is not None:
                    token_id = full_tokens[i]
                    # prompt_logprobs[i] is a dict mapping token_id -> Logprob object
                    if token_id in prompt_logprobs[i]:
                        response_logprobs.append(prompt_logprobs[i][token_id].logprob)
            
            # Sum all logprobs to get total log probability of response
            total_logprob = sum(response_logprobs) if response_logprobs else float('-inf')
            results.append(total_logprob)
    else:
        print('Judge model biased id: ', judge_model_biased_id, flush=True)
        # in text there are 5 neutral followed by 5 biased then again 5 neutral and 5 biased ...
        neutral_texts = []
        biased_texts = []
        neutral_response_texts = []
        biased_response_texts = []
        for i, (t, r) in enumerate(zip(texts, response_texts)):
            if i % 10 < 5: 
                neutral_texts.append(t)
                neutral_response_texts.append(r)
            else:
                biased_texts.append(t)
                biased_response_texts.append(r)
        neutral_res = []
        biased_res = []
        for prompt_text, response_text in zip(neutral_texts, neutral_response_texts):
            # Concatenate prompt and response
            full_text = prompt_text + response_text
            
            # Tokenize both parts separately to find boundary
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            
            # The response starts after the prompt tokens
            response_start_idx = len(prompt_tokens)
            # print('Text in neutral:', full_text, flush=True)
            # print('Len text in neutral:', len(full_tokens), flush=True)
            # print(len(full_tokens), flush=True)
            # Request prompt logprobs for all tokens
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,  # We only want prompt logprobs, not generation
                prompt_logprobs=len(full_tokens),  # Get logprobs for all prompt tokens
            )
            
            outputs = llm.generate([full_text], sampling_params, **lora_kwargs)
            output = outputs[0]
            
            # prompt_logprobs[i] contains logprob distribution for token at position i
            # We want logprobs for tokens from response_start_idx onward
            prompt_logprobs = output.prompt_logprobs
            
            # Aggregate logprobs for response portion
            response_logprobs = []
            for i in range(response_start_idx, len(full_tokens)):
                if prompt_logprobs[i] is not None:
                    token_id = full_tokens[i]
                    # prompt_logprobs[i] is a dict mapping token_id -> Logprob object
                    if token_id in prompt_logprobs[i]:
                        response_logprobs.append(prompt_logprobs[i][token_id].logprob)
            
            # Sum all logprobs to get total log probability of response
            total_logprob = sum(response_logprobs) if response_logprobs else float('-inf')
            neutral_res.append(total_logprob)
        
        biased_lora_kwargs = dict(lora_request=_build_lora_request(judge_model_biased_id))

        for prompt_text, response_text in zip(biased_texts, biased_response_texts):
            full_text = prompt_text + response_text
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            response_start_idx = len(prompt_tokens)
            # print('Text in biased:', full_text, flush=True)
            # print('Len text in biased:', len(full_tokens), flush=True)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                prompt_logprobs=len(full_tokens),
            )
            outputs = llm.generate([full_text], sampling_params, **biased_lora_kwargs)
            output = outputs[0]
            prompt_logprobs = output.prompt_logprobs
            response_logprobs = []
            for i in range(response_start_idx, len(full_tokens)):
                if prompt_logprobs[i] is not None:
                    token_id = full_tokens[i]
                    if token_id in prompt_logprobs[i]:
                        response_logprobs.append(prompt_logprobs[i][token_id].logprob)
            total_logprob = sum(response_logprobs) if response_logprobs else float('-inf')
            biased_res.append(total_logprob)
        # merge back results
        neutral_idx = 0
        biased_idx = 0
        for i in range(len(texts)):
            if i % 10 < 5:
                results.append(neutral_res[neutral_idx])
                neutral_idx += 1
            else:
                results.append(biased_res[biased_idx])
                biased_idx += 1
    return results