from dataclasses import field
from pydantic import BaseModel
from sl.llm.data_models import LLMResponse, SampleCfg, Judgment


class Evaluation(BaseModel):
    questions: list[str]
    n_samples_per_question: int
    sample_cfg: SampleCfg
    judgment_map: dict[str, Judgment] = field(default_factory=dict)


class EvaluationResponse(BaseModel):
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)


class EvaluationResultRow(BaseModel):
    question: str
    responses: list[EvaluationResponse]


class LogprobEvaluation(BaseModel):
    """Configuration for logprob-based evaluation."""
    questions: list[str]
    sample_cfg: SampleCfg


class LogprobEvaluationResultRow(BaseModel):
    """Result for a single question with full token logprobs."""
    question: str
    token_logprobs: dict[str, float]  # Full logprobs dict for first token
