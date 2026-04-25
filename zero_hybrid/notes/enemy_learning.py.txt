# 🔥 VERY LIGHT enemy damage inference

def apply_enemy_damage(blobs):
    for b in blobs:
        if hasattr(b, "cluster") and b.cluster >= 2:
            if not hasattr(b, "hp_estimate"):
                b.hp_estimate = 5

            b.hp_estimate = max(1, b.hp_estimate - 0.3)