import cv2
import numpy as np

from ocr_scoring_system import (
    OCRScoringSystem,
    MultiClassFishClassifier,
    BulletConservationManager
)

from enhanced_ai import (
    FishMemory,
    ShotTracker,
    AdaptiveStrategy
)

# =========================
# INIT SYSTEMS
# =========================
ocr_system = OCRScoringSystem()
fish_classifier = MultiClassFishClassifier()
bullet_manager = BulletConservationManager()

fish_memory = FishMemory()
shot_tracker = ShotTracker()
strategy = AdaptiveStrategy()

aggressiveness = 0.5

# =========================
# CONTROLS
# =========================
def adjust_aggressiveness(delta):
    global aggressiveness
    aggressiveness = min(max(0.0, aggressiveness + delta), 1.0)

def key_event_handler(key):
    if key == ord('+'):
        adjust_aggressiveness(0.1)
    elif key == ord('-'):
        adjust_aggressiveness(-0.1)

# =========================
# SIMPLE DETECTION (SAFE)
# =========================
def detect_fish(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fish_list = []

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 200:
            continue

        fish_list.append({
            'id': i,
            'area': area,
            'x': x,
            'y': y,
            'vx': 0,
            'vy': 0,
            'confidence': 0.8,
            'value': area / 100
        })

    return fish_list

# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fish_objects = detect_fish(frame)

    for fish in fish_objects:
        # classify
        class_id = fish_classifier.classify(fish)
        fish['class_id'] = class_id

        # learned multiplier (OCR system hook)
        multiplier, conf = ocr_system.get_expected_multiplier(class_id)
        value = fish['value'] * multiplier

        # confidence from shot history
        history_conf = shot_tracker.get_confidence_for_class(class_id) / 100.0

        # combine confidence sources
        combined_conf = max(fish['confidence'], history_conf)

        # memory prediction
        category, mem_conf = fish_memory.predict_killable(class_id, shots_available=3)

        # adaptive strategy gate (score placeholder = 50)
        if not strategy.should_fire(score=50):
            continue

        # final fire decision
        should_fire = bullet_manager.should_fire(
            target_value=value,
            confidence=combined_conf,
            aggression=aggressiveness
        )

        if should_fire:
            cost = strategy.adjust_shot_cost(0, 1)

            bullet_manager.record_shot(value, cost, hit=True)

            # record into trackers
            shot_tracker.record_shot(
                fish_id=fish['id'],
                fish_class=class_id,
                area=fish['area'],
                x=fish['x'],
                y=fish['y'],
                vx=fish['vx'],
                vy=fish['vy'],
                outcome='kill'
            )

            fish_memory.record_kill(class_id, shots_fired=1, value=value)

            strategy.reset_burst()

        # DRAW DEBUG
        x, y = fish['x'], fish['y']
        cv2.rectangle(frame, (x, y), (x+40, y+40), (0,255,0), 2)

        cv2.putText(frame,
            f"C{class_id} V:{int(value)}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0,255,0),
            1
        )

    # UI
    cv2.putText(frame,
        f"Agg:{aggressiveness:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.imshow("Fish AI", frame)

    key = cv2.waitKey(1)
    key_event_handler(key)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()