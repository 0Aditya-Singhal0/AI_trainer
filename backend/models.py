from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class FeedbackItem(BaseModel):
    error_type: str
    confidence: float
    message: str
    severity: SeverityEnum

class FormFeedback(BaseModel):
    timestamp: str
    exercise_type: str
    detected_errors: List[FeedbackItem]
    overall_score: float
    recommendations: List[str]

class VideoUploadResponse(BaseModel):
    status: str
    form_feedback: Optional[FormFeedback]
    processing_time: float

class ExercisePrediction(BaseModel):
    exercise_type: str
    confidence: float

class ExerciseClassificationResponse(BaseModel):
    status: str
    predictions: List[ExercisePrediction]
    processing_time: float