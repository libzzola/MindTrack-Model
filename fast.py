from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import time
import numpy as np
from collections import defaultdict
import pandas as pd

# Assuming the test engine code is already in a file `adaptive_test_engine.py` and imported here
from adaptive_test_engine import load_questions, AdaptiveTestEngine, DIFFICULTY_MAP, MODALITY_TYPES

# Initialize FastAPI app
app = FastAPI()

# Pydantic models for handling API input and output
class UserAnswer(BaseModel):
    user_answer: str

class TestResults(BaseModel):
    iq_score: float
    skill_performance: dict
    difficulty_distribution: dict

# Load questions (you should replace the file path with the actual one)
questions = load_questions("Example data.csv")
engine = AdaptiveTestEngine(questions)

@app.get("/start_test/")
async def start_test(background: str):
    """
    Starts a new test for a given background. Background options: 
    '1': Business, '2': Social Sciences, '3': CS & Math
    """
    if background not in ['1', '2', '3']:
        raise HTTPException(status_code=400, detail="Invalid background choice")
    
    # Start the test and return a status
    try:
        engine.run_test(background)
        return {"message": "Test started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting test: {str(e)}")

@app.get("/question/")
async def get_question():
    """
    Returns the next question. This is just an example, you might need to adjust it.
    """
    if not engine.question_history:
        raise HTTPException(status_code=400, detail="Test not started yet")
    
    # You can customize this to return a specific question, based on question index
    question = engine.question_history[-1]  # Get the most recent question
    return {
        "question": question['question'],
        "options": question['options'],
        "question_id": question['id']
    }

@app.post("/answer/{question_id}")
async def answer_question(question_id: int, user_answer: UserAnswer):
    """
    Submit an answer for a specific question.
    """
    # Look up the question by ID
    question = next((q for q in engine.questions if q['ID'] == question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Calculate the reward, score, and update engine
    start_time = time.time()
    correct = (user_answer.user_answer.lower() == question['Answer'].strip().lower())
    time_taken = time.time() - start_time
    
    reward = engine.assign_reward(correct, time_taken, question['Difficulty'])
    norm_reward = engine.normalize_reward(reward)
    engine.score += reward

    # Log results
    engine.question_history.append({
        'id': question['ID'],
        'skill': question['AssociatedSkill'],
        'modality': question['Type'],
        'difficulty': question['Difficulty'],
        'correct': correct,
        'time': time_taken,
        'reward': reward
    })

    # Return feedback
    return {
        "correct": correct,
        "reward": reward,
        "total_score": engine.score
    }

@app.get("/results/")
async def get_results():
    """
    Returns the final results of the test.
    """
    iq_score = engine.score  # Customize how you calculate IQ score based on your reward logic
    skill_performance = engine.skill_performance
    difficulty_distribution = defaultdict(int)

    # Calculate difficulty distribution
    for question in engine.question_history:
        difficulty_distribution[question['difficulty']] += 1
    
    return TestResults(
        iq_score=iq_score,
        skill_performance=skill_performance,
        difficulty_distribution=difficulty_distribution
    )

# To run the server, use Uvicorn (command below)
# uvicorn <filename>:app --reload

