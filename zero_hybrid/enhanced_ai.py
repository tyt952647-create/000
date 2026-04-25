import json
import time
import numpy as np
from collections import defaultdict, deque

# ===========================
# FISH MEMORY
# ===========================
class FishMemory:
    def __init__(self, storage_path='fish_memory.json'):
        self.storage_path = storage_path
        self.fish_classes = defaultdict(lambda: {
            'shots_to_kill': [],
            'value': 0,
            'encounters': 0,
            'avg_shots': 0.0
        })
        self.load()

    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump({k: dict(v) for k, v in self.fish_classes.items()}, f, indent=2)

    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    self.fish_classes[int(k)] = v
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def record_kill(self, fish_class, shots_fired, value):
        entry = self.fish_classes[fish_class]
        entry['shots_to_kill'].append(shots_fired)
        entry['value'] = value
        entry['encounters'] += 1
        entry['avg_shots'] = np.mean(entry['shots_to_kill']) if entry['shots_to_kill'] else 0
        self.save()

    def predict_killable(self, fish_class, shots_available):
        entry = self.fish_classes[fish_class]

        if not entry['shots_to_kill']:
            return None, 0.0

        avg = entry['avg_shots']

        if avg <= 3:
            return 'easy', 0.9
        elif avg > 50:
            return 'hard', 0.8
        else:
            return 'medium', 0.5


# ===========================
# SHOT TRACKER (FIXED)
# ===========================
class ShotTracker:
    def __init__(self, storage_path='shot_history.json'):
        self.storage_path = storage_path
        self.session_shots = []
        self.load()

    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.session_shots, f)

    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    self.session_shots = []
                else:
                    self.session_shots = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            self.session_shots = []

    def record_shot(self, fish_id, fish_class, area, x, y, vx, vy, outcome):
        shot = {
            'timestamp': time.time(),
            'fish_id': fish_id,
            'fish_class': fish_class,
            'area': area,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'outcome': outcome
        }
        self.session_shots.append(shot)
        self.save()

    def get_confidence_for_class(self, fish_class):
        shots = [s for s in self.session_shots if s['fish_class'] == fish_class]

        if not shots:
            return 0.0

        kills = len([s for s in shots if s['outcome'] == 'kill'])
        return (kills / len(shots)) * 100


# ===========================
# ADAPTIVE STRATEGY
# ===========================
class AdaptiveStrategy:
    def __init__(self, initial_shot_cost=1):
        self.base_cost = initial_shot_cost
        self.shot_cost = initial_shot_cost
        self.last_fire_time = time.time()
        self.score_threshold = 10

    def should_fire(self, score):
        if time.time() - self.last_fire_time > 60:
            return True
        return score > self.score_threshold

    def adjust_shot_cost(self, burst_index, burst_size):
        if burst_index >= 2:
            self.shot_cost = self.base_cost * (1.5 ** (burst_index - 1))
        else:
            self.shot_cost = self.base_cost
        return self.shot_cost

    def reset_burst(self):
        self.shot_cost = self.base_cost
        self.last_fire_time = time.time()