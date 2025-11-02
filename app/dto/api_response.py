from pydantic import BaseModel

class Metadata(BaseModel):
    request_id: str
    timestamp: str

class VerifyData(BaseModel):
    model_version: str
    is_me: bool
    score: float
    threshold: float
    timing_ms: float

class ApiResponse(BaseModel):
    success: bool
    data: VerifyData
    metadata: Metadata