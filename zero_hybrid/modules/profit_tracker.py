"""
Profit tracking module for session management
"""

import time
import json
import os

PROFIT_FILE = "profit_history.json"

last_score = 0
total_spent = 0
total_earned = 0
session_start = time.time()

def update_score(new_score):
    """Register score change"""
    global last_score, total_earned
    
    delta = new_score - last_score
    if delta > 0:
        total_earned += delta
    
    last_score = new_score
    return delta

def register_shot(cost):
    """Register a shot expense"""
    global total_spent
    total_spent += cost

def get_profit():
    """Get current profit/loss"""
    return total_earned - total_spent

def should_spend(bank):
    """Decision logic for spending bankroll"""
    profit = get_profit()
    
    if profit > 50:
        return "aggressive"
    elif profit > 0:
        return "normal"
    else:
        return "conservative"

def save_session():
    """Save session data"""
    elapsed = time.time() - session_start
    data = {
        "duration_seconds": elapsed,
        "total_earned": total_earned,
        "total_spent": total_spent,
        "profit": get_profit()
    }
    
    try:
        history = []
        if os.path.exists(PROFIT_FILE):
            with open(PROFIT_FILE, 'r') as f:
                history = json.load(f)
        
        history.append(data)
        with open(PROFIT_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass
