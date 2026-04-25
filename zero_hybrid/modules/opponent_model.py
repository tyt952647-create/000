class OpponentModel:
    def __init__(self):
        self.decay_rate = 0.05

    def update(self, blobs):
        for b in blobs:
            # initialize if missing
            if not hasattr(b, "hp_est"):
                b.hp_est = 5.0

            # cluster = likely multiple players hitting
            if getattr(b, "cluster", 1) >= 2:
                b.hp_est -= 0.3

            # fast disappear → high competition
            if b.area < 500:  # dying / small
                b.hp_est -= 0.2

            # decay floor
            b.hp_est = max(1.0, b.hp_est)

    def competition_score(self, b):
        # lower hp_est → more competition → higher priority
        return 1.0 / b.hp_est