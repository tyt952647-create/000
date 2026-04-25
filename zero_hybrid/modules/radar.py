import cv2
import numpy as np
import mss

class Radar:
    def __init__(self):
        self.sct = mss.mss()

        self.monitor = {
            "left": 0,
            "top": 0,
            "width": 800,
            "height": 600
        }

        self.threshold = 90
        self.min_area = 15

    def grab(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if w * h > self.min_area:
                detections.append((int(x), int(y), int(w), int(h)))

        return detections

    def show(self, frame, blobs, intersections):
        vis = frame.copy()

        for b in blobs:
            cv2.rectangle(vis, (b.x, b.y), (b.x+b.w, b.y+b.h), (0,255,0), 2)

        for (x, y) in intersections:
            cv2.circle(vis, (x, y), 5, (0,0,255), -1)

        cv2.imshow("radar", vis)

        return cv2.waitKey(1) & 0xFF == ord('q')