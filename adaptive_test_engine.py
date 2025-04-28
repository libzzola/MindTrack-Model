import numpy as np
import random
import time
import pandas as pd
from collections import defaultdict

# Configuration
DIFFICULTY_MAP = {
    "easy": 0, "mid": 1, "medium": 2,
    "hard": 3, "advanced": 4, "exceptional": 5
}
MODALITY_TYPES = ['textual', 'image', 'auditory']
MIN_QUESTIONS_PER_TYPE = 5
TOTAL_QUESTIONS = 15
DIFFICULTY_WEIGHTS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

def load_questions(file_path):
    """Load and validate questions from CSV"""
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Clean and standardize data
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
    df['Type'] = df['Type'].apply(lambda x: x if x in MODALITY_TYPES else 'text')

    return df.to_dict('records')

def validate_question_bank(questions):
    """Ensure we have sufficient questions"""
    print("\nQuestion Bank Validation:")
    validation_passed = True

    # Check modality distribution
    modality_counts = defaultdict(int)
    for q in questions:
        modality_counts[q['Type']] += 1

    for modality in MODALITY_TYPES:
        count = modality_counts.get(modality, 0)
        print(f"- {modality}: {count} questions")
        if count < MIN_QUESTIONS_PER_TYPE:
            print(f"  ⚠️ Needs at least {MIN_QUESTIONS_PER_TYPE} {modality} questions")
            validation_passed = False

    # Check difficulty distribution
    print("\nDifficulty Distribution:")
    difficulty_counts = defaultdict(int)
    for q in questions:
        difficulty_counts[q['Difficulty']] += 1

    for diff, name in sorted((v,k) for k,v in DIFFICULTY_MAP.items()):
        count = difficulty_counts.get(diff, 0)
        print(f"- {name}: {count} questions")
        if count < 3:
            print(f"  ⚠️ Needs more {name} questions")

    return validation_passed

class AdaptiveTestEngine:
    def __init__(self, questions):
        self.questions = questions
        self.used_questions = set()
        self.skill_performance = defaultdict(lambda: {'correct': 0, 'attempts': 0})
        self.question_history = []

        # Initialize UCB parameters
        self.diff_rewards = np.zeros(len(DIFFICULTY_MAP))
        self.diff_counts = np.zeros(len(DIFFICULTY_MAP))
        self.modality_rewards = np.zeros(len(MODALITY_TYPES))
        self.modality_counts = np.zeros(len(MODALITY_TYPES))

        self.total_trials = 0
        self.score = 0
        self.consecutive_correct = 0
        self.consecutive_wrong = 0

    def ucb_select(self, estimates, counts):
      """Performance-aware UCB selection"""
      self.total_trials += 1
      unexplored = np.where(counts == 0)[0]

      # If user is struggling (score < 0), prefer easier questions
      if self.score < 0 and len(unexplored) == 0:
          weights = np.array([1.0/(i+1) for i in range(len(estimates))])  # Prefer lower indices
          weighted_estimates = estimates * weights
          ucb_values = weighted_estimates + 1.5 * np.sqrt((2 * np.log(self.total_trials)) / counts)
          return np.argmax(ucb_values)

      # Normal UCB for neutral/positive scores
      if len(unexplored) > 0:
          return random.choice(unexplored)
      ucb_values = estimates + 2 * np.sqrt((2 * np.log(self.total_trials)) / counts)
      return np.argmax(ucb_values)

    def adjust_difficulty(self, current_diff, correct):
        """More conservative difficulty adjustment"""
        if correct:
            self.consecutive_correct += 1
            self.consecutive_wrong = 0

            # Only increase difficulty after 3 consecutive correct answers
            if self.consecutive_correct >= 3:
                new_diff = min(current_diff + 1, len(DIFFICULTY_MAP) - 1)
                # Never jump more than one level at a time
                if new_diff - current_diff <= 1:
                    return new_diff
        else:
            self.consecutive_wrong += 1
            self.consecutive_correct = 0

            # Decrease difficulty after just 1 wrong answer
            if self.consecutive_wrong >= 1:
                new_diff = max(current_diff - 1, 0)
                # Never drop more than one level at a time
                if current_diff - new_diff <= 1:
                    return new_diff

        return current_diff

    def normalize_reward(self, r):
        return max(0.0, min(1.0, (r + 1) / 3))

    def assign_reward(self, correct, time_taken, difficulty_idx):
        base = 1 if correct else -1
        time_penalty = min(time_taken / 10, 1)
        return (base - time_penalty) * DIFFICULTY_WEIGHTS[difficulty_idx]

    def update_estimates(self, estimates, counts, index, reward):
        counts[index] += 1
        n = counts[index]
        estimates[index] += (reward - estimates[index]) / n

    def select_question(self, modality_idx, difficulty_idx, preferred_skills):
        """Guaranteed question selection with strict modality"""
        modality = MODALITY_TYPES[modality_idx]
        candidates = [
            q for q in self.questions
            if q['Type'] == modality
            and str(q['ID']) not in self.used_questions
        ]

        # Try exact difficulty match first
        exact_diff = [q for q in candidates if q['Difficulty'] == difficulty_idx]
        if exact_diff:
            candidates = exact_diff

        # Filter by preferred skills if available
        if preferred_skills:
            skilled = [q for q in candidates if q.get('AssociatedSkill') in preferred_skills]
            if skilled:
                candidates = skilled

        if candidates:
            return random.choice(candidates)

        # Final fallback with warning
        candidates = [q for q in self.questions if str(q['ID']) not in self.used_questions]
        if candidates:
            print(f"⚠️ Using fallback {candidates[0]['Type']} question")
            return random.choice(candidates)
        return None

    def present_question(self, question, q_num):
        """Handle question presentation and scoring"""
        print(f"\nQuestion {q_num+1}/{TOTAL_QUESTIONS} (Score: {self.score:.2f})")
        print(f"Mode: {question['Type'].capitalize()} | Difficulty: {list(DIFFICULTY_MAP.keys())[question['Difficulty']].capitalize()}")
        print(f"Question: {question['Question']}")

        if pd.notna(question.get('Media')) and str(question.get('Media')).strip():
            print(f"Media: {question['Media']}")

        # Process options
        options = [opt.strip() for opt in str(question['Options']).split(',') if opt.strip()]
        if question['Answer'].strip() not in options:
            options.append(question['Answer'].strip())
        options = list(set(options))
        random.shuffle(options)

        option_letters = "ABCDEF"[:len(options)]
        option_map = {letter: opt for letter, opt in zip(option_letters, options)}

        # Display options
        for letter, opt in option_map.items():
            print(f"{letter}. {opt}")

        # Get and validate user answer
        while True:
            user_answer = input(f"Your answer ({'/'.join(option_map.keys())}): ").strip().upper()
            if user_answer in option_map:
                break
            print("Invalid choice. Please enter one of:", '/'.join(option_map.keys()))

        return option_map, user_answer

    def run_test(self, background):
        background_skills = {
            '1': ["Listening & Comprehension", "General Knowledge"],
            '2': ["Abstract Reasoning", "Verbal Reasoning"],
            '3': ["Mathematical Reasoning", "Spatial Reasoning"]
        }.get(background, [])

        for q_num in range(TOTAL_QUESTIONS):
            # Select modality and difficulty
            modality_idx = self.ucb_select(self.modality_rewards, self.modality_counts)
            difficulty_idx = self.ucb_select(self.diff_rewards, self.diff_counts)

            # Get question
            question = self.select_question(modality_idx, difficulty_idx, background_skills)
            if not question:
                print("❌ Failed to find suitable question")
                break

            self.used_questions.add(str(question['ID']))

            # Present question and get answer
            start_time = time.time()
            option_map, user_answer = self.present_question(question, q_num)
            time_taken = time.time() - start_time

            # Check answer
            correct = (user_answer in option_map and
                      option_map[user_answer].lower() == question['Answer'].strip().lower())

            # Calculate reward
            reward = self.assign_reward(correct, time_taken, question['Difficulty'])
            norm_reward = self.normalize_reward(reward)
            self.score += reward

            # Update performance tracking
            skill = question.get('AssociatedSkill')
            if skill:
                self.skill_performance[skill]['attempts'] += 1
                if correct:
                    self.skill_performance[skill]['correct'] += 1

            # Update UCB models
            self.update_estimates(self.diff_rewards, self.diff_counts, question['Difficulty'], norm_reward)
            self.update_estimates(self.modality_rewards, self.modality_counts, modality_idx, norm_reward)

            # Adjust difficulty
            difficulty_idx = self.adjust_difficulty(difficulty_idx, correct)

            # Log results
            print("✅ Correct!" if correct else "❌ Incorrect.")
            print(f"Time: {time_taken:.1f}s | Reward: {reward:.2f} | Score: {self.score:.2f}")

            self.question_history.append({
                'id': question['ID'],
                'skill': skill,
                'modality': question['Type'],
                'difficulty': question['Difficulty'],
                'correct': correct,
                'time': time_taken,
                'reward': reward
            })

        self.display_results()

    def display_results(self):
        """Show final test results"""
        raw_scaled = 100 + (self.score * 2)
        iq_score = max(70, min(130, raw_scaled))
        print(f"\nYour Estimated IQ Score: {iq_score:.0f}")

        print("\n📊 Skill Performance Analysis:")
        for skill, data in sorted(self.skill_performance.items(),
                                key=lambda x: x[1]['correct']/x[1]['attempts'],
                                reverse=True):
            accuracy = (data['correct'] / data['attempts']) * 100
            print(f"- {skill}: {accuracy:.1f}% ({data['correct']}/{data['attempts']})")

        print("\n🔍 Difficulty Distribution:")
        for diff in sorted(set(q['difficulty'] for q in self.question_history)):
            count = sum(1 for q in self.question_history if q['difficulty'] == diff)
            correct = sum(1 for q in self.question_history if q['difficulty'] == diff and q['correct'])
            print(f"- {list(DIFFICULTY_MAP.keys())[diff].capitalize()}: {correct}/{count} correct")

if __name__ == "__main__":
    questions = load_questions("Example data.csv")
    if not validate_question_bank(questions):
        print("\n⚠️ Warning: Question bank may be insufficient for optimal testing")

    print("\nWelcome to the Fully Adaptive IQ Test")
    print("Select your background:")
    print("1. Business\n2. Social Sciences\n3. CS & Math")
    background = input("Enter 1-3: ").strip()

    engine = AdaptiveTestEngine(questions)
    engine.run_test(background)