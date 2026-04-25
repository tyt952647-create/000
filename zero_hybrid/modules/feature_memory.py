import json
import os

FILE = "data/features.json"

memory = []

def load():
    global memory
    if os.path.exists(FILE):
        try:
            memory = json.load(open(FILE))
        except:
            memory = []

def save():
    os.makedirs("data", exist_ok=True)
    json.dump(memory, open(FILE, "w"))

def add_feature(blob):
    feat = {
        "area": blob.area,
        "aspect": blob.w / max(1, blob.h),
        "speed": abs(blob.vx) + abs(blob.vy)
    }
    memory.append(feat)

load()