import cv2
import numpy as np
import pytesseract
import json
import time
from collections import defaultdict, deque
from pathlib import Path

class OCRScoringSystem:
    """Extract and learn score multipliers from game OCR"""
    
    def __init__(self, storage_path='ocr_learning.json'):
        self.storage_path = storage_path
        self.ocr_data = defaultdict(lambda: {
            'detections': [],
            'multipliers': [],
            'avg_multiplier': 1.0,
            'confidence': 0.0,
            'encounters': 0
        })
        self.load()
    
    def detect_score_text(self, frame, roi=None):
        """Extract score/multiplier text from frame using OCR"""
        try:
            if roi:
                x, y, w, h = roi
                crop = frame[y:y+h, x:x+w]
            else:
                crop = frame
            
            # Preprocess for OCR
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # OCR detection
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except:
            return ""
    
    def extract_multiplier(self, text):
        """Parse multiplier from OCR text"""
        multipliers = {'DOUBLE': 2.0, 'TRIPLE': 3.0, '2X': 2.0, '3X': 3.0}
        
        for key, val in multipliers.items():
            if key in text.upper():
                return val
        
        # Try numeric extraction
        import re
        matches = re.findall(r'(\d+)(?:\s*[Xx×])?', text)
        if matches:
            try:
                return float(matches[0])
            except:
                pass
        
        return 1.0
    
    def record_kill(self, fish_class, multiplier):
        """Record kill with detected multiplier"""
        entry = self.ocr_data[fish_class]
        entry['multipliers'].append(multiplier)
        entry['encounters'] += 1
        entry['avg_multiplier'] = np.mean(entry['multipliers']) if entry['multipliers'] else 1.0
        entry['confidence'] = min(100, (len(entry['multipliers']) / max(entry['encounters'], 1)) * 100)
        self.save()
    
    def get_expected_multiplier(self, fish_class):
        """Get learned multiplier for fish class"""
        entry = self.ocr_data[fish_class]
        return entry['avg_multiplier'], entry['confidence'] / 100.0
    
    def save(self):
        with open(self.storage_path, 'w') as f:
            data = {k: dict(v) for k, v in self.ocr_data.items()}
            json.dump(data, f, indent=2)
    
    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    self.ocr_data[int(k) if k.isdigit() else k] = v
        except FileNotFoundError:
            pass


class MultiClassFishClassifier:
    """Classify fish into 10+ dynamic classes"""
    
    def __init__(self):
        self.classes = {
            0: {'name': 'Tiny', 'size_range': (0, 100), 'color': (0, 255, 0)},
            1: {'name': 'Small', 'size_range': (100, 300), 'color': (50, 255, 0)},
            2: {'name': 'Medium', 'size_range': (300, 600), 'color': (0, 255, 255)},
            3: {'name': 'Large', 'size_range': (600, 1200), 'color': (0, 128, 255)},
            4: {'name': 'Huge', 'size_range': (1200, 2500), 'color': (0, 0, 255)},
            5: {'name': 'Boss', 'size_range': (2500, 5000), 'color': (255, 0, 0)},
            6: {'name': 'Speed-Low', 'speed_range': (0, 1), 'color': (200, 200, 0)},
            7: {'name': 'Speed-Med', 'speed_range': (1, 3), 'color': (200, 100, 0)},
            8: {'name': 'Speed-High', 'speed_range': (3, 10), 'color': (200, 0, 0)},
            9: {'name': 'Armored', 'armor_marker': True, 'color': (128, 128, 128)},
        }
    
    def classify(self, fish):
        """Multi-attribute fish classification"""
        classes = []
        
        # Size-based
        area = fish.get('area', 0)
        for cid, info in self.classes.items():
            if 'size_range' in info:
                min_size, max_size = info['size_range']
                if min_size <= area < max_size:
                    classes.append(cid)
        
        # Speed-based
        speed = fish.get('speed', 0)
        for cid, info in self.classes.items():
            if 'speed_range' in info:
                min_speed, max_speed = info['speed_range']
                if min_speed <= speed < max_speed:
                    classes.append(cid)
        
        # Special markers
        if fish.get('armor_marker', False):
            classes.append(9)
        
        # Primary class = smallest/most specific
        return min(classes) if classes else 0
    
    def get_class_info(self, class_id):
        return self.classes.get(class_id, {'name': 'Unknown', 'color': (255, 255, 255)})


class BulletConservationManager:
    """Track ammo efficiency and optimize shot selection"""
    
    def __init__(self, cost_per_shot=1.0):
        self.cost_per_shot = cost_per_shot
        self.total_shots = 0
        self.total_cost = 0.0
        self.total_gain = 0.0
        self.shot_history = deque(maxlen=1000)
        self.efficiency_threshold = 1.5
        self.last_shot_time = time.time()
        self.min_fire_interval = 0.5
    
    def should_fire(self, target_value, confidence=1.0, aggression=0.5):
        """Determine if shot is worth firing"""
        # Aggression multiplies confidence
        adjusted_confidence = confidence * (0.5 + aggression)
        expected_gain = target_value * adjusted_confidence
        total_cost = self.cost_per_shot
        
        # Must exceed efficiency threshold
        threshold = self.efficiency_threshold * (2.0 - aggression)  # Aggressive = lower threshold
        if expected_gain < total_cost * threshold:
            return False
        
        # Respect minimum fire interval
        if time.time() - self.last_shot_time < self.min_fire_interval:
            return False
        
        return True
    
    def record_shot(self, target_value, cost, hit=True):
        """Track shot outcome"""
        self.total_shots += 1
        self.total_cost += cost
        if hit:
            self.total_gain += target_value
        
        self.shot_history.append({
            'value': target_value,
            'cost': cost,
            'hit': hit,
            'efficiency': target_value / cost if cost > 0 else 0,
            'timestamp': time.time()
        })
        self.last_shot_time = time.time()
    
    def get_efficiency_ratio(self):
        """Overall shots-to-gain ratio"""
        if self.total_cost == 0:
            return 0
        return self.total_gain / self.total_cost
    
    def get_recent_efficiency(self, window=100):
        """Recent efficiency trend"""
        recent = list(self.shot_history)[-window:]
        if not recent:
            return 0
        total_value = sum(s['value'] for s in recent)
        total_cost = sum(s['cost'] for s in recent)
        return total_value / total_cost if total_cost > 0 else 0
    
    def adjust_efficiency_threshold(self, recent_ratio):
        """Auto-adjust efficiency requirement"""
        if recent_ratio < 1.2:
            self.efficiency_threshold = max(1.0, self.efficiency_threshold - 0.1)
        elif recent_ratio > 2.0:
            self.efficiency_threshold = min(3.0, self.efficiency_threshold + 0.1)