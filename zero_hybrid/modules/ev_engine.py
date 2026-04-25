import json
import os
import time
from collections import defaultdict

FILE = "data/ev_memory.json"

class EVEngine:
    def __init__(self):
        self.class_value = defaultdict(lambda: 1.0)

        self.last_score = 0
        self.total_spent = 0
        self.total_earned = 0

        self.shots = []
        self.load()

    # =========================
    # PERSISTENCE
    # =========================
    def load(self):
        if os.path.exists(FILE):
            try:
                data = json.load(open(FILE))
                self.class_value.update(data.get("class_value", {}))
            except:
                pass

    def save(self):
        os.makedirs("data", exist_ok=True)
        json.dump({
            "class_value": dict(self.class_value)
        }, open(FILE, "w"))

    # =========================
    # LEARNING FROM SCORE
    # =========================
    def update_score(self, new_score):
        delta = new_score - self.last_score

        if delta > 0:
            self.total_earned += delta
            now = time.time()

            # reward recent shots
            for t, cls, cost in self.shots[-10:]:
                if now - t < 2.0:
                    self.class_value[cls] += delta * 0.1

        self.last_score = new_score

    def register_shot(self, cls, cost=1):
        self.total_spent += cost
        self.shots.append((time.time(), cls, cost))

    # =========================
    # DECISION (USES LEARNING)
    # =========================
    def score_blob(self, b, opponent):
        # learned value
        learned = self.class_value[b.cls]

        # smaller = better
        size_score = 1.0 / max(1, b.area)

        # opponent competition
        comp = opponent.competition_score(b)

        return learned * size_score * 1000 * (1 + comp)

    def best_small_target(self, blobs, opponent):
        small = [b for b in blobs if b.cls == 0]
        if not small:
            return None

        return max(small, key=lambda b: self.score_blob(b, opponent))

    def get_profit(self):
        return self.total_earned - self.total_spent