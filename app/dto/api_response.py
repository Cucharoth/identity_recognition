from pydantic import BaseModel

class Metadata(BaseModel):
    request_id: str
    timestamp: str

class VerifyData(BaseModel):
    is_me: bool
    score: float
    threshold: float
    timing_ms: float

class ApiResponse(BaseModel):
    model_version: str
    data: VerifyData
    metadata: Metadata