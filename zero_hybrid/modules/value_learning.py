import json, os

FILE = "data_memory.json"
memory = {}

def load():
    global memory
    if os.path.exists(FILE):
        try:
            memory.update(json.load(open(FILE)))
        except:
            memory.clear()

def save():
    json.dump(memory, open(FILE,"w"))

load()

def bucket(area):
    if area < 2000: return "small"
    elif area < 5000: return "medium"
    else: return "large"

def estimate_value(area):
    b = bucket(area)
    return memory.get(b, {"small":2,"medium":5,"large":15}[b])

def update_value(area, gain, shots):
    b = bucket(area)
    val = gain / max(1, shots)

    old = memory.get(b, val)
    memory[b] = (old*0.8)+(val*0.2)
    save()