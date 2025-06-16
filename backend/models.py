from pydantic import BaseModel
from typing import List, Optional


class ExtractedText(BaseModel):
    content: str
    confidence: Optional[float] = None
    bounding_box: Optional[List[float]] = None


class MenuScanResult(BaseModel):
    extracted_text: List[ExtractedText]
    raw_text: str
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None
