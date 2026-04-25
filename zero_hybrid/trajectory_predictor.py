"""
Trajectory prediction and lead calculation for moving targets
"""

import numpy as np
from collections import deque

class TrajectoryPredictor:
    """Predict fish future position for accurate aiming"""
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.position_history = deque(maxlen=lookback)
        self.velocity_smoother = deque(maxlen=5)
    
    def update(self, x, y, timestamp=None):
        """Update trajectory history"""
        self.position_history.append((x, y, timestamp))
    
    def predict(self, frames_ahead=3):
        """Predict position N frames ahead"""
        if len(self.position_history) < 2:
            if self.position_history:
                return self.position_history[-1][0], self.position_history[-1][1]
            return 0, 0
        
        # Get recent positions
        positions = list(self.position_history)
        x1, y1, t1 = positions[-2]
        x2, y2, t2 = positions[-1]
        
        # Calculate velocity
        dt = 1.0 if t2 is None or t1 is None else max(t2 - t1, 0.001)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        # Smooth velocity
        self.velocity_smoother.append((vx, vy))
        avg_vx = np.mean([v[0] for v in self.velocity_smoother])
        avg_vy = np.mean([v[1] for v in self.velocity_smoother])
        
        # Predict
        pred_x = x2 + avg_vx * frames_ahead
        pred_y = y2 + avg_vy * frames_ahead
        
        return pred_x, pred_y
    
    def get_velocity(self):
        """Get current smoothed velocity"""
        if not self.velocity_smoother:
            return 0, 0
        avg_vx = np.mean([v[0] for v in self.velocity_smoother])
        avg_vy = np.mean([v[1] for v in self.velocity_smoother])
        return avg_vx, avg_vy
    
    def get_speed(self):
        """Get current speed magnitude"""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)
