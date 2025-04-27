import numpy as np
import random
import time
import pandas as pd
import os
from collections import defaultdict

DIFFICULTY_MAP = {
    "easy": 0, "mid": 1, "medium": 2,
    "hard": 3, "advanced": 4, "exceptional": 5
}
DIFFICULTY_LIST = list(DIFFICULTY_MAP.keys())
MODALITY_TYPES = ['textual', 'image', 'auditory']
MODALITY_IDX = {k: i for i, k in enumerate(MODALITY_TYPES)}

def load_questions(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')  # Fixed encoding
    df = df.rename(columns=lambda x: x.strip())
    df = df.rename(columns={
        'ï»¿Question ID': 'ID',
        'Question Text': 'Question',
        'Difficulty Level': 'Difficulty',
        'Correct Answer': 'Answer',
        'Skill Sets': 'Skill',
        'Question Type': 'Type',
        'Answer Options': 'Options',
        'media_url': 'Media',
        'Associated Skill': 'AssociatedSkill'
    })
    df['Difficulty'] = df['Difficulty'].str.lower().map(DIFFICULTY_MAP)
    df = df.dropna(subset=['Difficulty'])
    df['Difficulty'] = df['Difficulty'].astype(int)
    df['Type'] = df['Type'].str.lower().str.strip()
    df['Type'] = df['Type'].apply(lambda x: x if x in MODALITY_TYPES else 'textual')
    return df.to_dict('records')

class AdaptiveIQTest:
    def __init__(self, user_id, questions, background):
        self.user_id = user_id
        self.questions = questions
        self.background = background
        self.used_questions = set()
        self.question_history = []
        self.score = 0

        # Initialize rewards with background bias
        self.modality_rewards = np.zeros(len(MODALITY_TYPES))
        self.modality_counts = np.zeros(len(MODALITY_TYPES))
        self.difficulty_rewards = np.zeros(len(DIFFICULTY_MAP))
        self.difficulty_counts = np.zeros(len(DIFFICULTY_MAP))
        
        # Apply background bias
        bg_config = {
            '1': (2, 0.4),  # Business: boost medium difficulty
            '2': (1, 0.6),  # Social Sciences: boost mid
            '3': (3, 0.3)   # CS/Math: boost hard
        }.get(background, (2, 0))
        self.difficulty_rewards[bg_config[0]] += bg_config[1]

        self.total_trials = 0
        self.load_past_attempts()

    def load_past_attempts(self):
        if os.path.exists("user_attempts.csv"):
            df = pd.read_csv("user_attempts.csv")
            past = df[df['user_id'] == self.user_id]
            if not past.empty:
                for _, row in past.iterrows():
                    mod_idx = MODALITY_IDX[row['modality']]
                    diff = row['difficulty']
                    reward = row['reward']
                    self.update_estimates(self.modality_rewards, self.modality_counts, mod_idx, reward)
                    self.update_estimates(self.difficulty_rewards, self.difficulty_counts, diff, reward)

    def _adjust_difficulty_bias(self, current_diff):
        total = sum(1 for q, _ in self.question_history if q['Difficulty'] == current_diff)
        if total < 3:
            return
        correct = sum(1 for q, correct in self.question_history if q['Difficulty'] == current_diff and correct)
        success_rate = correct / total
        
        if success_rate < 0.4 and current_diff > 0:
            self.difficulty_rewards[current_diff-1] += 0.2
        elif success_rate > 0.7 and current_diff < len(DIFFICULTY_MAP)-1:
            self.difficulty_rewards[current_diff+1] += 0.2

    def ucb_select(self, estimates, counts):
        self.total_trials += 1
        if (counts == 0).any():
            return random.choice(np.where(counts == 0)[0])
        return np.argmax(estimates + 1.5 * np.sqrt(np.log(self.total_trials + 1) / (counts + 1e-5)))

    def update_estimates(self, estimates, counts, index, reward):
        counts[index] += 1
        estimates[index] += (reward - estimates[index]) / counts[index]

    def normalize_reward(self, r):
        return max(-1.0, min(1.0, r / 3))  # Preserve negatives

    def assign_reward(self, correct, time_taken, difficulty):
        base = 1.5 if correct else -1.5
        time_penalty = min(time_taken / 8, 1.2)
        return (base - time_penalty) * (1.0 + 0.15 * difficulty)

    def select_question(self):
        m_idx = self.ucb_select(self.modality_rewards, self.modality_counts)
        d_idx = self.ucb_select(self.difficulty_rewards, self.difficulty_counts)
        modality = MODALITY_TYPES[m_idx]
        filtered = [q for q in self.questions if q['Type'] == modality and str(q['ID']) not in self.used_questions]
        exact = [q for q in filtered if q['Difficulty'] == d_idx]
        
        if exact:
            question = random.choice(exact)
        elif filtered:
            question = random.choice(filtered)
        else:
            remaining = [q for q in self.questions if str(q['ID']) not in self.used_questions]
            if remaining:
                question = random.choice(remaining)
            else:
                return None  # No more questions available

        self.used_questions.add(str(question['ID']))
        return question


    def process_answer(self, question_id, user_answer, time_taken):
        q = next((q for q in self.questions if str(q['ID']) == str(question_id)), None)
        if not q:
            return False, 0.0, 0.0
        
        correct = user_answer.strip().lower() == q['Answer'].strip().lower()
        reward = self.assign_reward(correct, time_taken, q['Difficulty'])
        norm_reward = self.normalize_reward(reward)
        
        self.used_questions.add(str(q['ID']))
        self.score += int(correct)
        self.question_history.append((q, correct))
        
        # Update estimates
        self.update_estimates(
            self.modality_rewards, 
            self.modality_counts, 
            MODALITY_IDX[q['Type']], 
            norm_reward
        )
        self.update_estimates(
            self.difficulty_rewards,
            self.difficulty_counts,
            q['Difficulty'],
            norm_reward
        )
        self._adjust_difficulty_bias(q['Difficulty'])
        
        return correct, reward, norm_reward

    def get_results(self):
        iq_score = 100 + (self.score * 1.5)
        return {
            "estimated_iq": int(np.clip(iq_score, 70, 140)),
            "score": float(self.score),
            "answered": len(self.question_history)
        }