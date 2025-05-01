from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import numpy as np
import random
import time
import uuid
from collections import defaultdict

# Create FastAPI app
app = FastAPI(title="Adaptive IQ Test API", 
              description="API for serving an adaptive IQ test based on reinforcement learning")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DIFFICULTY_MAP = {
    "easy": 0, "mid": 1, "medium": 2,
    "hard": 3, "advanced": 4, "exceptional": 5
}
DIFFICULTY_LIST = list(DIFFICULTY_MAP.keys())
MODALITY_TYPES = ['textual', 'image', 'auditory']
MODALITY_IDX = {k: i for i, k in enumerate(MODALITY_TYPES)}

# Pydantic models for request/response validation
class Question(BaseModel):
    id: int
    modality: str
    difficulty: str

class PreviousStatistics(BaseModel):
    easy: float = 50
    mid: float = 50
    medium: float = 50
    hard: float = 50
    advanced: float = 50
    exceptional: float = 50
    textual: float = 50
    image: float = 50
    auditory: float = 50

class StartGameRequest(BaseModel):
    user_id: str
    background: int = Field(..., ge=1, le=3)
    question_pool: List[Question]
    question_count: int = Field(..., ge=1)
    previous_statistics: Optional[PreviousStatistics] = None

class NextQuestionRequest(BaseModel):
    previousQuestionId: int
    sessionId: str
    correct: bool

class GameSession(BaseModel):
    session_id: str
    user_id: str
    background: int
    question_pool: List[Dict[str, Any]]
    question_count: int
    current_question_index: int = 0
    question_history: List[Dict[str, Any]] = []
    used_questions: List[int] = []
    score: int = 0
    total_trials: int = 0
    modality_rewards: List[float] = []
    modality_counts: List[int] = []
    difficulty_rewards: List[float] = []
    difficulty_counts: List[int] = []
    modality_perf: Dict[str, Dict[str, int]] = {}
    diff_perf: Dict[str, Dict[str, int]] = {}
    
    class Config:
        arbitrary_types_allowed = True

# In-memory storage (replace with database in production)
active_sessions = {}

# Helper functions
def normalize_reward(r):
    """Normalize reward to [-1, 1] range"""
    return max(-1.0, min(1.0, r / 3))

def assign_reward(correct, difficulty):
    """Calculate reward based on correctness and difficulty"""
    base = 1 if correct else -1
    return base * (1.0 + 0.1 * DIFFICULTY_MAP.get(difficulty, 0))

def ucb_select(estimates, counts, total_trials):
    """UCB algorithm for selection"""
    if (counts == 0).any():
        return random.choice(np.where(counts == 0)[0])
    return np.argmax(estimates + 2 * np.sqrt((2 * np.log(total_trials)) / counts))

def update_estimates(estimates, counts, index, reward):
    """Update UCB estimates"""
    counts[index] += 1
    estimates[index] += (reward - estimates[index]) / counts[index]

def adjust_difficulty_bias(session, current_difficulty):
    """Dynamically adjust difficulty rewards based on performance"""
    diff_perf = session.diff_perf.get(current_difficulty, {'correct': 0, 'total': 0})
    correct = diff_perf.get('correct', 0)
    total = diff_perf.get('total', 0)

    if total == 0:
        return

    success_rate = correct / total
    diff_idx = DIFFICULTY_MAP.get(current_difficulty, 0)

    # Boost adjacent difficulties based on performance
    if success_rate < 0.4 and diff_idx > 0:
        session.difficulty_rewards[diff_idx-1] += 0.15
    elif success_rate > 0.75 and diff_idx < len(DIFFICULTY_LIST)-1:
        session.difficulty_rewards[diff_idx+1] += 0.15

def select_question(session):
    """Select next question using UCB algorithm"""
    # Convert lists to numpy arrays for UCB calculations
    modality_rewards = np.array(session.modality_rewards)
    modality_counts = np.array(session.modality_counts)
    difficulty_rewards = np.array(session.difficulty_rewards)
    difficulty_counts = np.array(session.difficulty_counts)
    
    m_idx = ucb_select(modality_rewards, modality_counts, session.total_trials)
    d_idx = ucb_select(difficulty_rewards, difficulty_counts, session.total_trials)
    
    modality = MODALITY_TYPES[m_idx]
    difficulty = DIFFICULTY_LIST[d_idx]
    
    # Filter questions by modality and difficulty that haven't been used
    filtered = [q for q in session.question_pool 
                if q['modality'] == modality 
                and q['id'] not in session.used_questions]
    
    exact = [q for q in filtered if q['difficulty'] == difficulty]
    
    # Choose a question
    if exact:
        return random.choice(exact)
    elif filtered:
        return random.choice(filtered)
    else:
        # If all questions of the selected modality are used, pick any unused question
        unused = [q for q in session.question_pool if q['id'] not in session.used_questions]
        return random.choice(unused) if unused else None

def initialize_session(request_data: StartGameRequest):
    """Initialize a new game session"""
    session_id = str(uuid.uuid4())
    
    # Initialize arrays for UCB algorithm
    modality_rewards = [0.0] * len(MODALITY_TYPES)
    modality_counts = [0] * len(MODALITY_TYPES)
    difficulty_rewards = [0.0] * len(DIFFICULTY_MAP)
    difficulty_counts = [0] * len(DIFFICULTY_MAP)
    
    # Apply background bias to difficulty rewards
    difficulty_boost = {
        1: 0.2,  # Business: slight boost to medium difficulty
        2: 0.4,  # Social Sciences: stronger boost to easier questions
        3: -0.1  # CS/Math: harder questions get initial boost
    }.get(request_data.background, 0)
    
    # Apply to medium difficulty
    difficulty_rewards[2] += difficulty_boost
    
    # Initialize performance tracking dictionaries
    modality_perf = {mod: {'correct': 0, 'total': 0} for mod in MODALITY_TYPES}
    diff_perf = {diff: {'correct': 0, 'total': 0} for diff in DIFFICULTY_LIST}
    
    # Apply previous statistics if provided
    if request_data.previous_statistics:
        stats = request_data.previous_statistics
        
        # Update modality rewards based on previous performance
        for mod in MODALITY_TYPES:
            success_rate = getattr(stats, mod) / 100.0  # Convert percentage to proportion
            idx = MODALITY_IDX[mod]
            modality_rewards[idx] = 2 * (success_rate - 0.5)  # Convert to [-1,1] range
        
        # Update difficulty rewards based on previous performance
        for diff in DIFFICULTY_LIST:
            success_rate = getattr(stats, diff) / 100.0  # Convert percentage to proportion
            idx = DIFFICULTY_MAP[diff]
            weight = 1 + (0.1 * idx)  # Harder levels weighted more
            difficulty_rewards[idx] = 2 * (success_rate - 0.5) * weight
    
    # Create session
    session = GameSession(
        session_id=session_id,
        user_id=request_data.user_id,
        background=request_data.background,
        question_pool=[q.dict() for q in request_data.question_pool],
        question_count=request_data.question_count,
        modality_rewards=modality_rewards,
        modality_counts=modality_counts,
        difficulty_rewards=difficulty_rewards,
        difficulty_counts=difficulty_counts,
        modality_perf=modality_perf,
        diff_perf=diff_perf
    )
    
    # Store session
    active_sessions[session_id] = session
    
    return session

def process_answer(session, correct, question):
    """Process the user's answer and update the model"""
    session.total_trials += 1
    session.score += 1 if correct else 0
    
    # Update performance tracking
    modality = question['modality']
    difficulty = question['difficulty']
    
    # Update modality performance
    if modality not in session.modality_perf:
        session.modality_perf[modality] = {'correct': 0, 'total': 0}
    session.modality_perf[modality]['total'] += 1
    session.modality_perf[modality]['correct'] += int(correct)
    
    # Update difficulty performance
    if difficulty not in session.diff_perf:
        session.diff_perf[difficulty] = {'correct': 0, 'total': 0}
    session.diff_perf[difficulty]['total'] += 1
    session.diff_perf[difficulty]['correct'] += int(correct)
    
    # Calculate and apply reward
    reward = assign_reward(correct, difficulty)
    norm_reward = normalize_reward(reward)
    
    # Convert lists to numpy arrays
    modality_rewards = np.array(session.modality_rewards)
    modality_counts = np.array(session.modality_counts)
    difficulty_rewards = np.array(session.difficulty_rewards)
    difficulty_counts = np.array(session.difficulty_counts)
    
    # Update estimates
    update_estimates(modality_rewards, modality_counts, MODALITY_IDX[modality], norm_reward)
    update_estimates(difficulty_rewards, difficulty_counts, DIFFICULTY_MAP[difficulty], norm_reward)
    
    # Convert back to lists
    session.modality_rewards = modality_rewards.tolist()
    session.modality_counts = modality_counts.tolist()
    session.difficulty_rewards = difficulty_rewards.tolist()
    session.difficulty_counts = difficulty_counts.tolist()
    
    # Adjust difficulty bias
    adjust_difficulty_bias(session, difficulty)
    
    return session

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Adaptive IQ Test API"}

@app.post("/start_game")
async def start_game(request: StartGameRequest):
    """
    Start a new game session with the provided question pool
    """
    session = initialize_session(request)
    
    # Select the first question
    question = select_question(session)
    if not question:
        raise HTTPException(status_code=400, detail="No suitable questions available in the pool")
    
    # Mark question as used
    session.used_questions.append(question['id'])
    session.current_question_index += 1
    
    # Add to question history
    session.question_history.append({
        'question': question,
        'index': session.current_question_index
    })
    
    # Update session
    active_sessions[session.session_id] = session
    
    return {
        "session_id": session.session_id,
        "current_question": question,
        "question_number": session.current_question_index,
        "total_questions": session.question_count
    }

@app.post("/next_question")
async def next_question(request: NextQuestionRequest):
    """
    Process the previous answer and provide the next question
    """
    # Validate session exists
    if request.sessionId not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[request.sessionId]
    
    # Validate previous question
    last_question_info = session.question_history[-1] if session.question_history else None
    if not last_question_info or last_question_info['question']['id'] != request.previousQuestionId:
        raise HTTPException(status_code=400, detail="Invalid previous question ID")
    
    # Update the model based on previous answer
    process_answer(
        session=session,
        correct=request.correct,
        question=last_question_info['question']
    )
    
    # Check if we've reached the end of the test
    if session.current_question_index >= session.question_count:
        return calculate_results(session)
    
    # Select next question
    question = select_question(session)
    if not question:
        # If we run out of questions, end the test early
        return calculate_results(session)
    
    # Mark question as used
    session.used_questions.append(question['id'])
    session.current_question_index += 1
    
    # Add to question history
    session.question_history.append({
        'question': question,
        'index': session.current_question_index
    })
    
    # Update session
    active_sessions[session.session_id] = session
    
    return {
        "session_id": session.session_id,
        "current_question": question,
        "question_number": session.current_question_index,
        "total_questions": session.question_count
    }

def calculate_results(session):
    """Calculate and return test results"""
    # Calculate IQ score (simplified)
    iq_score = 100 + (session.score * 1.5)
    iq_score = max(70, min(140, iq_score))
    
    # Calculate performance metrics
    modality_performance = {}
    for mod in MODALITY_TYPES:
        if mod in session.modality_perf and session.modality_perf[mod]['total'] > 0:
            correct = session.modality_perf[mod]['correct']
            total = session.modality_perf[mod]['total']
            modality_performance[mod] = {
                "correct": correct,
                "total": total,
                "percentage": round((correct/total) * 100, 1)
            }
    
    difficulty_performance = {}
    for diff in DIFFICULTY_LIST:
        if diff in session.diff_perf and session.diff_perf[diff]['total'] > 0:
            correct = session.diff_perf[diff]['correct']
            total = session.diff_perf[diff]['total']
            difficulty_performance[diff] = {
                "correct": correct,
                "total": total,
                "percentage": round((correct/total) * 100, 1)
            }
    
    # Clean up the session (in a real application, you might want to store it in a database instead)
    del active_sessions[session.session_id]
    
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "completed": True,
        "score": session.score,
        "iq_estimate": int(iq_score),
        "questions_answered": session.current_question_index,
        "modality_performance": modality_performance,
        "difficulty_performance": difficulty_performance
    }

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Get information about an active session
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "current_question_index": session.current_question_index,
        "total_questions": session.question_count,
        "score": session.score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
