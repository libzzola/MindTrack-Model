from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import AdaptiveIQTest, load_questions  # Importing your model and functions
import uuid
import os
import json

app = FastAPI()

# Load questions globally (assuming questions are loaded once and used throughout)
QUESTIONS = load_questions("Example data.csv")

# Store sessions in memory (or use a database for production)
sessions = {}

class StartGameRequest(BaseModel):
    background: int  # Background: 1, 2, or 3
    question_pool: list  # List of questions (question IDs)
    question_count: int  # Number of questions to ask
    previous_statistics: float = 100.0  # Default is 100% for first-time users

class NextQuestionRequest(BaseModel):
    session_id: str
    previous_correct: bool  # Boolean value of whether the previous answer was correct

class GameSession:
    def __init__(self, user_id, background, question_pool, question_count, previous_statistics):
        self.user_id = user_id
        self.background = background
        self.test = AdaptiveIQTest(user_id, question_pool)
        self.test.background_bias = {1: 0, 2: 1, 3: 2}.get(str(background), 1)
        self.test.load_past_attempts()  # Load previous stats (if available)
        self.test.score = previous_statistics  # Initialize with the previous statistics as score
        self.session_id = str(uuid.uuid4())  # Generate a unique session ID
        self.question_history = []  # Initialize empty question history

    def get_next_question(self, correct: bool):
        self.test.adaptive_phase()  # Generate next question in adaptive phase
        return self.test.select_question()  # Get next question from the model

@app.post("/start_game/")
async def start_game(request: StartGameRequest):
    user_id = str(uuid.uuid4())  # Generate a unique user ID for each session
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    
    # Create a new game session
    session = GameSession(
        user_id=user_id,
        background=request.background,
        question_pool=QUESTIONS,
        question_count=request.question_count,
        previous_statistics=request.previous_statistics
    )

    # Store the session in the sessions dictionary
    sessions[session_id] = session
    
    # Return the session_id
    return {"session_id": session_id}

@app.post("/next_question/")
async def next_question(request: NextQuestionRequest):
    # Check if session exists
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get the next question
    next_q = session.get_next_question(correct=request.previous_correct)
    
    # If no question is left, end the game
    if next_q is None:
        return {"message": "Game over", "score": session.test.score}
    
    # Prepare the question and answer choices to send to the user
    question_data = {
        "question_id": next_q['ID'],
        "question_text": next_q['Question'],
        "type": next_q['Type'],
        "difficulty": DIFFICULTY_LIST[next_q['Difficulty']],
        "options": [opt.strip() for opt in next_q['Options'].split(',')]
    }
    
    return {"question": question_data}
