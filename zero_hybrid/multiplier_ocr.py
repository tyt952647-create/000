"""
OCR-based score multiplier detection and learning
Reads score changes on screen to determine fish value multipliers
"""

import cv2
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

class ScoreOCRReader:
    """Extract score numbers from game screen"""
    def __init__(self):
        self.score_regions = [
            (10, 10, 150, 40),      # Top-left
            (0, 0, 200, 50),        # Top-center  
            (-200, 10, -10, 40),    # Top-right
            (10, -50, 150, -10),    # Bottom-left
            (-200, -50, -10, -10),  # Bottom-right
        ]
        self.prev_score = 0
        self.score_history = defaultdict(list)
    
    def read_score(self, frame, region_idx=None):
        """OCR score from specific region"""
        h, w = frame.shape[:2]
        
        if region_idx is None:
            # Try all regions
            for idx, (x1, y1, x2, y2) in enumerate(self.score_regions):
                score = self._read_region(frame, x1, y1, x2, y2, w, h)
                if score is not None:
                    return score, idx
            return None, None
        else:
            x1, y1, x2, y2 = self.score_regions[region_idx]
            score = self._read_region(frame, x1, y1, x2, y2, w, h)
            return score, region_idx
    
    def _read_region(self, frame, x1, y1, x2, y2, w, h):
        """Extract number from region"""
        # Handle negative coordinates
        if x1 < 0:
            x1 = w + x1
        if x2 < 0:
            x2 = w + x2
        if y1 < 0:
            y1 = h + y1
        if y2 < 0:
            y2 = h + y2
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        # Preprocess for OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Simple digit extraction (without pytesseract dependency)
        # In production, you would use pytesseract here
        return None
    
    def detect_multiplier_from_kill(self, score_before, score_after, shot_cost=1.0):
        """Calculate multiplier from score change"""
        delta = score_after - score_before
        if delta <= 0:
            return None
        
        multiplier = delta / shot_cost
        return multiplier


class FishValueLearner:
    """Learn and track fish value multipliers by class"""
    def __init__(self, storage='fish_values.json'):
        self.storage = storage
        self.class_values = defaultdict(lambda: {
            'multipliers': [],
            'avg': 1.0,
            'confidence': 0.0,
            'encounters': 0
        })
        self.load()
    
    def record_kill(self, fish_class, multiplier):
        """Record observed multiplier for class"""
        entry = self.class_values[str(fish_class)]
        entry['multipliers'].append(multiplier)
        entry['encounters'] += 1
        
        # Calculate running average
        entry['avg'] = np.mean(entry['multipliers'][-50:])  # Last 50 kills
        
        # Confidence: 0 at 1 kill, 1 at 10+ kills
        entry['confidence'] = min(1.0, entry['encounters'] / 10.0)
        
        self.save()
    
    def get_expected_value(self, fish_class, base_value=1.0):
        """Get expected value with confidence"""
        entry = self.class_values[str(fish_class)]
        expected = base_value * entry['avg']
        return expected, entry['confidence']
    
    def save(self):
        try:
            with open(self.storage, 'w') as f:
                json.dump(dict(self.class_values), f, indent=2)
        except:
            pass
    
    def load(self):
        try:
            with open(self.storage, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    self.class_values[k] = v
        except:
            pass
