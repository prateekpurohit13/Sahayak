import pydantic
from typing import Optional, List

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: List[str]
    sources: List[str]


class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int