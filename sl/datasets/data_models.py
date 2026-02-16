from pydantic import BaseModel


class DatasetRow(BaseModel):
    prompt: str
    completion: str

class PreferenceDatasetRow(BaseModel):
    prompt: str
    response_a: str
    response_b: str
    preferred_response: str
    judge_reasoning: str | None = None
    preferred_response_swapped: str | None = None
    judge_reasoning_swapped: str | None = None
    is_consistent: bool = False

class PreferenceDatasetRowDeep(BaseModel):
    prompt: str
    response_a: str
    response_b: str
    response_c: str
    response_d: str
    response_e: str
    ratio_a: float | None = None
    ratio_b: float | None = None
    ratio_c: float | None = None
    ratio_d: float | None = None
    ratio_e: float | None = None
    preferred_response: str
    dispreferred_response: str
