# =========================
# COLOR CLASS LEARNING
# =========================

color_memory = {}  # color_id → multiplier

def classify_color(hsv):
    h,s,v = hsv

    # 🔥 many buckets (you can tune later)
    if h < 10: return "red"
    elif h < 25: return "orange"
    elif h < 35: return "yellow"
    elif h < 70: return "green"
    elif h < 110: return "cyan"
    elif h < 140: return "blue"
    elif h < 170: return "purple"
    else: return "pink"

def set_multiplier(color, mult):
    color_memory[color] = mult

def get_multiplier(color):
    return color_memory.get(color, 1)