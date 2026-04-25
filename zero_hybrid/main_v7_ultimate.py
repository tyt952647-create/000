"""
ZERO HYBRID v7.0 ULTIMATE - COMPLETE INTEGRATION
Motion-based fish detection with dual-screenshot synchronization,
advanced AI, adaptive learning, and always-on aggression meter
"""

import cv2
import numpy as np
import mss
import time
import pyautogui
import json
from collections import deque
from pathlib import Path

# Import all custom modules
from blob_synchronizer import BlobSynchronizationSystem
from radar_with_legend import RadarWithLegend
from aggression_overlay import AggressionOverlay
from enhanced_ai import FishMemory, ShotTracker, AdaptiveStrategy
from advanced_predictor import AdvancedPredictorAI
from adaptive_learning import AdaptiveLearningSystem
from failsafe_system import IntegratedFailsafeSystem
from ocr_scoring_system import OCRScoringSystem, MultiClassFishClassifier, BulletConservationManager
from priority_selector import PrioritySelector

# ===========================
# WINDOW POSITIONING
# ===========================
class WindowManager:
    """Manages separate GUI windows with proper positioning"""
    def __init__(self):
        self.windows = {}
        screen_w, screen_h = pyautogui.size()
        self.screen_w = screen_w
        self.screen_h = screen_h
        
    def create(self, name, x, y, width=500, height=500):
        """Create and position a window"""
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)
        cv2.moveWindow(name, x, y)
        self.windows[name] = (x, y, width, height)
        return name
    
    def show(self, name, image):
        """Display image in window"""
        if image is not None and image.size > 0:
            cv2.imshow(name, image)


# ===========================
# CHROME WINDOW DETECTION
# ===========================
def get_chrome_region():
    """Auto-detect Chrome window and get right-half region"""
    try:
        import win32gui
        import win32con
        
        def enum_windows(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and 'chrome' in win32gui.GetWindowText(hwnd).lower():
                results.append(hwnd)
        
        results = []
        win32gui.EnumWindows(enum_windows, results)
        
        if results:
            hwnd = results[0]
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top
            
            # Capture right-half only
            return {
                "left": left + width // 2,
                "top": top,
                "width": width // 2,
                "height": height
            }
    except:
        pass
    
    # Fallback: use primary monitor's right half
    sct = mss.mss()
    monitor = sct.monitors[1]
    return {
        "left": monitor["width"] // 2,
        "top": 0,
        "width": monitor["width"] // 2,
        "height": monitor["height"]
    }


# ===========================
# MOTION DETECTION ENGINE
# ===========================
class MotionDetector:
    """Detects only moving pixels, ignores static background and oscillations"""
    
    def __init__(self, width, height, adapt_rate=5):
        self.width = width
        self.height = height
        self.adapt_rate = adapt_rate
        self.prev_frame = None
        self.motion_history = deque(maxlen=60)
        self.base_threshold = 15
        self.target_brightness = 127
        self.current_brightness = 127
        self.oscillation_suppression = 0.0
        
    def update_adapt_rate(self, rate):
        """Update adaptation speed (0-100)"""
        self.adapt_rate = max(0.1, rate)
    
    def detect_motion(self, frame):
        """Frame differencing for motion-only detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray.copy()
        
        _, motion_mask = cv2.threshold(diff, self.base_threshold, 255, cv2.THRESH_BINARY)
        
        motion_pixels = np.count_nonzero(motion_mask)
        self.motion_history.append(motion_pixels)
        
        return motion_mask
    
    def detect_oscillation(self):
        """Detect 0.5-2 sec oscillating patterns (UI overlays)"""
        if len(self.motion_history) < 30:
            return False
        
        recent = list(self.motion_history)
        
        for period in range(15, 61):
            if len(recent) < period * 2:
                continue
            
            first_half = recent[:period]
            second_half = recent[period:period*2]
            
            if len(first_half) == len(second_half):
                corr = np.corrcoef(first_half, second_half)[0, 1]
                if corr > 0.85:
                    self.oscillation_suppression = 1.0
                    return True
        
        self.oscillation_suppression = max(0, self.oscillation_suppression - 0.05)
        return False
    
    def adapt_threshold(self, motion_mask):
        """Auto-adapt threshold based on motion brightness"""
        motion_pixels = motion_mask > 0
        
        if np.any(motion_pixels):
            self.current_brightness = np.mean(self.prev_frame[motion_pixels])
            delta = (self.target_brightness - self.current_brightness) * (self.adapt_rate / 100.0)
            self.base_threshold = max(5, self.base_threshold + delta)


# ===========================
# BLOB TRACKER & CLUSTERING
# ===========================
class Fish:
    """Represents a single fish target"""
    def __init__(self, x, y, w, h, fish_id):
        self.id = fish_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.vx = 0
        self.vy = 0
        self.speed = 0
        self.heading = 0
        self.class_id = self._classify()
        self.history = deque(maxlen=10)
        self.history.append((x, y))
        self.age = 0
        self.last_seen = time.time()
        self.shots = 0
        self.sync_verified = False
        self.sync_confidence = 0.0
        
    def _classify(self):
        """Classify fish by size into 10 classes"""
        if self.area < 50:
            return 0
        elif self.area < 100:
            return 1
        elif self.area < 200:
            return 2
        elif self.area < 400:
            return 3
        elif self.area < 600:
            return 4
        elif self.area < 900:
            return 5
        elif self.area < 1200:
            return 6
        elif self.area < 1800:
            return 7
        elif self.area < 2500:
            return 8
        else:
            return 9
    
    def update(self, x, y, w, h):
        """Update position and velocity"""
        self.vx = x - self.x
        self.vy = y - self.y
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        
        if self.speed > 0:
            self.heading = np.arctan2(self.vy, self.vx) * 180 / np.pi
        
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.class_id = self._classify()
        self.history.append((x, y))
        self.age += 1
        self.last_seen = time.time()
    
    def is_same_trajectory(self, other, dist_thresh=30, angle_thresh=15, speed_thresh=0.5):
        """Check if two blobs are same fish"""
        dist = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        if dist > dist_thresh:
            return False
        
        if self.speed < 2 or other.speed < 2:
            return dist < dist_thresh * 0.5
        
        angle_diff = abs(self.heading - other.heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > angle_thresh:
            return False
        
        speed_ratio = min(self.speed, other.speed) / max(self.speed, other.speed, 0.1)
        if speed_ratio < (1.0 - speed_thresh):
            return False
        
        return True


class BlobTracker:
    """Tracks blobs and merges same-trajectory fragments into single fish"""
    
    def __init__(self):
        self.fishes = []
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections, motion_mask):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        valid_detections = []
        for x, y, w, h in detections:
            if w < 3 or h < 3:
                continue
            
            cx, cy = x + w // 2, y + h // 2
            if motion_mask[min(cy, motion_mask.shape[0]-1), min(cx, motion_mask.shape[1]-1)] > 128:
                valid_detections.append((x, y, w, h))
        
        matched = set()
        
        for x, y, w, h in valid_detections:
            best_match = None
            best_dist = float('inf')
            
            for fish in self.fishes:
                dist = np.sqrt((fish.x - x)**2 + (fish.y - y)**2)
                if dist < best_dist and dist < 50:
                    best_match = fish
                    best_dist = dist
            
            if best_match:
                best_match.update(x, y, w, h)
                matched.add(best_match.id)
            else:
                new_fish = Fish(x, y, w, h, self.next_id)
                self.fishes.append(new_fish)
                self.next_id += 1
        
        self.fishes = [f for f in self.fishes if f.id in matched or f.age < 5]
        self._merge_trajectories()
        
        return self.fishes
    
    def _merge_trajectories(self):
        """Merge fish with identical trajectory into one"""
        merged = []
        used = set()
        
        for i, f1 in enumerate(self.fishes):
            if f1.id in used:
                continue
            
            group = [f1]
            used.add(f1.id)
            
            for j, f2 in enumerate(self.fishes):
                if j <= i or f2.id in used:
                    continue
                
                if f1.is_same_trajectory(f2):
                    group.append(f2)
                    used.add(f2.id)
            
            group_sorted = sorted(group, key=lambda f: f.area, reverse=True)
            keeper = group_sorted[0]
            
            for absorbed in group_sorted[1:]:
                keeper.shots += absorbed.shots
            
            merged.append(keeper)
        
        self.fishes = merged


# ===========================
# MAIN APPLICATION
# ===========================
def main():
    print("🎮 ZERO HYBRID v7.0 ULTIMATE - COMPLETE INTEGRATION")
    print("=" * 70)
    print("✨ Features:")
    print("  • Dual-screenshot blob synchronization")
    print("  • Shape-based fish classification (10 classes)")
    print("  • Radar with integrated legend")
    print("  • Always-on aggression meter overlay")
    print("  • Advanced AI: convergence detection, splash damage")
    print("  • Adaptive learning (6 modules)")
    print("  • Failsafe system (6+6 protocols)")
    print("  • OCR multiplier learning")
    print("  • Autofire toggle (Shift key)")
    print("=" * 70)
    
    # Setup
    region = get_chrome_region()
    print(f"📍 Capturing from: {region}")
    
    sct = mss.mss()
    wm = WindowManager()
    motion_detector = MotionDetector(region["width"], region["height"])
    tracker = BlobTracker()
    
    # Initialize all AI systems
    blob_sync = BlobSynchronizationSystem(region)
    radar_legend = RadarWithLegend()
    aggression_overlay = AggressionOverlay()
    
    ocr_system = OCRScoringSystem()
    classifier = MultiClassFishClassifier()
    bullet_manager = BulletConservationManager(cost_per_shot=1.0)
    
    advanced_ai = AdvancedPredictorAI()
    adaptive_learning = AdaptiveLearningSystem()
    failsafe_system = IntegratedFailsafeSystem()
    
    fish_memory = FishMemory()
    shot_tracker = ShotTracker()
    priority_selector = PrioritySelector(fish_memory, shot_tracker)
    
    failsafe_system.start()
    
    # Create windows
    wm.create("VISUALIZATION", 50, 50, 900, 700)
    wm.create("RADAR_LEGEND", 1000, 50, 590, 420)
    wm.create("STATS", 1000, 500, 590, 250)
    
    # Control parameters
    fire_aggression = 0.5
    
    print("🎯 Motion-based detection active")
    print("🔄 Blob synchronization enabled")
    print("⚙️ All AI systems loaded")
    print("🔴 Controls: Q=quit, +=aggressive, -=conservative, Shift=autofire toggle")
    print("=" * 70)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture
            screenshot = np.array(sct.grab(region))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            if frame is None or frame.size == 0:
                print("⚠️ Frame capture failed")
                time.sleep(0.1)
                continue
            
            h, w = frame.shape[:2]
            
            # Motion detection
            motion_mask = motion_detector.detect_motion(frame)
            
            if motion_mask is None or motion_mask.size == 0:
                continue
            
            is_oscillating = motion_detector.detect_oscillation()
            
            if is_oscillating:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            motion_detector.adapt_threshold(motion_mask)
            
            # Find contours
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for c in contours:
                x, y, w_c, h_c = cv2.boundingRect(c)
                area = w_c * h_c
                
                if area < 20:
                    continue
                
                aspect = w_c / max(h_c, 1)
                if aspect > 5 or aspect < 0.2:
                    continue
                
                detections.append((x, y, w_c, h_c))
            
            # Blob synchronization (every 0.25s)
            qualified_blobs = blob_sync.process_frame(frame, motion_mask)
            
            # Track blobs
            fishes = tracker.update(detections, motion_mask)
            
            # Update qualified blob info into fish objects
            for fish in fishes:
                for qblob in qualified_blobs:
                    if abs(fish.x - qblob['x']) < 20 and abs(fish.y - qblob['y']) < 20:
                        fish.sync_verified = qblob.get('verified', False)
                        fish.sync_confidence = qblob.get('sync_quality', 0.0)
                        fish.class_id = qblob.get('class_id', fish.class_id)
                        break
            
            # Advanced AI processing
            ai_result = advanced_ai.process_frame(fishes, 0)
            
            # Adaptive learning update
            frame_data = {
                'fish_count': len(fishes),
                'motion_pixels': np.count_nonzero(motion_mask),
                'fps': frame_count / (time.time() - start_time + 0.001),
                'detection_success': len(qualified_blobs) / max(len(fishes), 1)
            }
            adaptive_learning.update_all_systems(frame_data)
            
            # === VISUALIZATION ===
            vis = frame.copy()
            
            class_colors = [
                (0, 255, 0),      # Tiny
                (50, 255, 0),     # Very Small
                (100, 255, 0),    # Small
                (150, 255, 0),    # Small-Medium
                (0, 255, 255),    # Medium
                (0, 200, 255),    # Medium-Large
                (0, 128, 255),    # Large
                (0, 100, 200),    # Very Large
                (0, 0, 255),      # Huge
                (255, 0, 0),      # Colossal
            ]
            
            for fish in fishes:
                color = class_colors[min(fish.class_id, 9)]
                cv2.rectangle(vis, (fish.x, fish.y), (fish.x + fish.w, fish.y + fish.h), color, 2)
                
                cx, cy = fish.x + fish.w // 2, fish.y + fish.h // 2
                sync_color = (0, 255, 0) if fish.sync_verified else (0, 0, 255)
                cv2.circle(vis, (cx, cy), 5, sync_color, -1)
                
                if fish.speed > 2:
                    px = cx + int(fish.vx * 3)
                    py = cy + int(fish.vy * 3)
                    cv2.arrowedLine(vis, (cx, cy), (px, py), (255, 0, 0), 2, tipLength=0.3)
                
                label = f"C:{fish.class_id} Sync:{fish.sync_confidence:.1f}"
                cv2.putText(vis, label, (fish.x, fish.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Stats
            fps = frame_count / (time.time() - start_time + 0.001)
            stats = f"FPS:{fps:.1f} | Targets:{len(fishes)} | Sync:{len(qualified_blobs)}"
            cv2.putText(vis, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Composite aggression overlay
            vis = aggression_overlay.composite_on_frame(vis, position='top-right', opacity=0.85)
            
            wm.show("VISUALIZATION", vis)
            
            # === RADAR WITH LEGEND ===
            radar_display = radar_legend.create_display(fishes, w, h)
            wm.show("RADAR_LEGEND", radar_display)
            
            # === STATS ===
            stats_display = np.zeros((250, 590, 3), dtype=np.uint8)
            stats_display[:] = (20, 20, 20)
            
            y_offset = 30
            cv2.putText(stats_display, f"SYSTEM STATS", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 40
            
            cv2.putText(stats_display, f"FPS: {fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(stats_display, f"Total Targets: {len(fishes)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(stats_display, f"Synchronized: {len(qualified_blobs)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(stats_display, f"Sync Quality: {blob_sync.sync_stats['avg_sync_quality']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(stats_display, f"Data Quality: {advanced_ai.corruption_handler.get_data_quality():.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            wm.show("STATS", stats_display)
            
            # === AGGRESSION METER UPDATE ===
            aggression_overlay.update_state(
                aggression_level=int(fire_aggression * 200),
                forcing_active=False,
                autofire_enabled=advanced_ai.autofire_manager.auto_fire_enabled,
                threshold=0.7
            )
            
            # === INPUT HANDLING ===
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('+') or key == ord('='):
                fire_aggression = min(1.0, fire_aggression + 0.1)
                print(f"📈 Aggression: {fire_aggression:.0%}")
            elif key == ord('-') or key == ord('_'):
                fire_aggression = max(0.0, fire_aggression - 0.1)
                print(f"📉 Aggression: {fire_aggression:.0%}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        failsafe_system.stop()
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")


if __name__ == "__main__":
    main()
