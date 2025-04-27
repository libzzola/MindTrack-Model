from pydantic import BaseModel
from typing import List, Optional

class StartTest(BaseModel):
    user_id: str
    background: str  # '1', '2', or '3'

class SubmitAnswer(BaseModel):
    question_id: str
    user_answer: str
    time_taken: float

class QuestionOut(BaseModel):
    question_id: str
    question: str
    options: List[str]
    type: str
    difficulty: int
    media: Optional[str] = None

class AnswerResult(BaseModel):
    correct: bool
    reward: float
    normalized_reward: float

class ResultSummary(BaseModel):
    estimated_iq: int
    score: float  # Matches engine's float conversion
    answered: int