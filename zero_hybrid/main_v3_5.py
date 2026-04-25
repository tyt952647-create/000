"""
ZERO HYBRID v3.5 - Complete Advanced AI System
All systems integrated:
- Motion detection
- Fish classification
- Convergence prediction
- Multi-shot strategy
- Adaptive learning (6 systems)
- Failsafe & emergency protocols
"""

import cv2
import numpy as np
import mss
import time
import pyautogui
from collections import deque
import threading

from advanced_predictor import AdvancedPredictorAI
from adaptive_learning import AdaptiveLearningSystem
from failsafe_system import IntegratedFailsafeSystem

# ===========================
# MOTION DETECTION
# ===========================
class MotionDetector:
    """Pure motion-only detection"""
    
    def __init__(self, adapt_rate=5):
        self.prev_frame = None
        self.motion_history = deque(maxlen=60)
        self.base_threshold = 15
        self.adapt_rate = adapt_rate
        
    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray.copy()
        
        _, motion_mask = cv2.threshold(diff, self.base_threshold, 255, cv2.THRESH_BINARY)
        self.motion_history.append(np.count_nonzero(motion_mask))
        
        return motion_mask

# ===========================
# FISH TRACKER
# ===========================
class Fish:
    def __init__(self, x, y, w, h, fish_id):
        self.id = fish_id
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.area = w * h
        self.vx, self.vy = 0, 0
        self.speed = 0
        self.heading = 0
        self.class_id = self._classify()
        self.age = 0
        self.last_seen = time.time()
        self.shots = 0
    
    def _classify(self):
        if self.area < 100: return 0
        elif self.area < 300: return 1
        elif self.area < 600: return 2
        elif self.area < 1200: return 3
        else: return 4
    
    def update(self, x, y, w, h):
        self.vx = x - self.x
        self.vy = y - self.y
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        if self.speed > 0:
            self.heading = np.arctan2(self.vy, self.vx) * 180 / np.pi
        self.x, self.y, self.w, self.h = x, y, w, h
        self.area = w * h
        self.class_id = self._classify()
        self.age += 1
        self.last_seen = time.time()

class BlobTracker:
    def __init__(self):
        self.fishes = []
        self.next_id = 0
    
    def update(self, detections, motion_mask):
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
                self.fishes.append(Fish(x, y, w, h, self.next_id))
                self.next_id += 1
        
        self.fishes = [f for f in self.fishes if f.id in matched or f.age < 5]
        return self.fishes

# ===========================
# MAIN APPLICATION
# ===========================
def main():
    print("🎮 ZERO HYBRID v3.5 - COMPLETE AI SYSTEM")
    print("=" * 70)
    print("✨ Systems loaded:")
    print("   ✅ Motion detection (frame differencing)")
    print("   ✅ Fish classification (5 classes)")
    print("   ✅ Convergence prediction (2+ small fish)")
    print("   ✅ Multi-shot strategy (3-shot bursts)")
    print("   ✅ Adaptive learning (6 modules)")
    print("   ✅ Failsafe system (6 protocols)")
    print("   ✅ Shift-key toggle control")
    print("=" * 70)
    
    # Initialize systems
    sct = mss.mss()
    monitor = sct.monitors[1]
    region = {
        "left": monitor["width"] // 2,
        "top": 0,
        "width": monitor["width"] // 2,
        "height": monitor["height"]
    }
    
    motion_detector = MotionDetector()
    tracker = BlobTracker()
    
    # AI Systems
    predictor = AdvancedPredictorAI()
    learning = AdaptiveLearningSystem()
    failsafe = IntegratedFailsafeSystem()
    
    # Start failsafe
    failsafe.start()
    
    # Metrics
    frame_count = 0
    start_time = time.time()
    shots_fired = 0
    score = 100
    score_history = deque(maxlen=200)
    
    print("\n🟢 Starting main loop...")
    print("⌨️ Press Shift to toggle auto-fire")
    print("🔴 Press Q to quit\n")
    
    try:
        while True:
            screenshot = np.array(sct.grab(region))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            if frame is None or frame.size == 0:
                time.sleep(0.01)
                continue
            
            h, w = frame.shape[:2]
            
            # Motion detection
            motion_mask = motion_detector.detect_motion(frame)
            
            # Find contours
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for c in contours:
                x, y, w_rect, h_rect = cv2.boundingRect(c)
                area = w_rect * h_rect
                if area < 20:
                    continue
                aspect = w_rect / max(h_rect, 1)
                if aspect > 5 or aspect < 0.2:
                    continue
                detections.append((x, y, w_rect, h_rect))
            
            # Track fish
            fishes = tracker.update(detections, motion_mask)
            
            # Run AI predictions
            predictor_result = predictor.process_frame(fishes, score)
            
            # Update learning systems
            frame_data = {
                'fish_classes': [f.class_id for f in fishes],
                'fps': frame_count / (time.time() - start_time + 0.001),
                'data_quality': predictor_result['data_quality'],
                'score': score,
                'score_history': list(score_history)
            }
            learning.update_all_systems(frame_data)
            
            # Run safety checks
            object_counts = {
                'fishes': len(fishes),
                'motions': np.count_nonzero(motion_mask)
            }
            
            failsafe_data = {
                'fps': frame_data['fps'],
                'data_quality': predictor_result['data_quality'],
                'score': score,
                'score_history': list(score_history),
                'object_counts': object_counts
            }
            
            failsafe_result = failsafe.update_all_checks(failsafe_data)
            
            # ===== FIRING DECISION =====
            should_fire = False
            if predictor_result['should_fire'] and failsafe_result['safe_to_fire']:
                if failsafe_result['autofire_enabled']:
                    collision = predictor_result['collisions'][0] if predictor_result['collisions'] else None
                    if collision:
                        should_fire = True
                        target_x, target_y = collision['collision_point']
                        
                        # Fire 3-shot burst
                        for burst_idx in range(3):
                            pyautogui.click(region["left"] + target_x, region["top"] + target_y)
                            shots_fired += 1
                            shot_cost = predictor.multi_shot.get_shot_cost(burst_idx)
                            score -= shot_cost
                            score_history.append(score)
                            
                            if burst_idx < 2:
                                time.sleep(0.03)
            
            # ===== VISUALIZATION =====
            vis = frame.copy()
            
            # Draw fish
            for fish in fishes:
                color = (0, 255, 0) if fish.speed > 2 else (100, 100, 100)
                cv2.rectangle(vis, (fish.x, fish.y), (fish.x + fish.w, fish.y + fish.h), color, 2)
                cx, cy = fish.x + fish.w // 2, fish.y + fish.h // 2
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                
                if fish.speed > 2:
                    px = cx + int(fish.vx * 3)
                    py = cy + int(fish.vy * 3)
                    cv2.arrowedLine(vis, (cx, cy), (px, py), (255, 0, 0), 2, tipLength=0.3)
            
            # Draw convergence points
            for collision in predictor_result['collisions']:
                cx, cy = collision['collision_point']
                cv2.circle(vis, (cx, cy), 20, (0, 165, 255), 2)
                cv2.putText(vis, f"T{collision['collision_frame']}", (cx, cy - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # Stats
            fps = frame_count / (time.time() - start_time + 0.001)
            failsafes_str = f"({len(failsafe_result['active_failsafes'])})" if failsafe_result['active_failsafes'] else ""
            status = "🎯 FIRING" if should_fire else "⏸️ Ready"
            
            cv2.putText(vis, f"FPS:{fps:.1f} | Targets:{len(fishes)} | Score:{score:.1f} | {status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"AutoFire:{failsafe_result['autofire_state']} | Shots:{shots_fired} | {failsafes_str}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if failsafe_result['safe_to_fire'] else (0, 0, 255), 2)
            
            cv2.imshow("ZERO HYBRID v3.5", vis)
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    
    except Exception as e:
        print(f"\n🚨 Error: {e}")
        failsafe.emergency_recovery.log_error("main_loop", str(e))
        if failsafe.emergency_recovery.attempt_recovery():
            print("Retrying...")
        else:
            print("Recovery failed")
    
    finally:
        failsafe.stop()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("📊 FINAL STATS")
        print(f"🎯 Score: {score:.1f}")
        print(f"💥 Shots: {shots_fired}")
        print(f"🐟 Detected: {len(fishes)}")
        print(f"⏱️ Time: {(time.time() - start_time):.1f}s")
        print(f"📡 FPS Avg: {frame_count / (time.time() - start_time + 0.001):.1f}")
        print("\n" + "=" * 70)
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main()
