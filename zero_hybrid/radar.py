import cv2
import numpy as np
import mss

class Radar:
    def __init__(self):
        self.sct = mss.mss()

        monitor = self.sct.monitors[1]

        # RIGHT HALF
        self.monitor = {
            "left": monitor["width"] // 2,
            "top": 0,
            "width": monitor["width"] // 2,
            "height": monitor["height"]
        }

        self.threshold = 85
        self.min_area = 20
        self.kernel_size = 5

        cv2.namedWindow("RADAR")
        cv2.namedWindow("MASK")

        cv2.createTrackbar("Threshold", "MASK", 85, 255, lambda x: None)
        cv2.createTrackbar("MinArea", "MASK", 20, 500, lambda x: None)
        cv2.createTrackbar("Kernel", "MASK", 5, 15, lambda x: None)

    def grab(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect(self, frame):
        # sliders
        self.threshold = cv2.getTrackbarPos("Threshold", "MASK")
        self.min_area = cv2.getTrackbarPos("MinArea", "MASK")
        k = max(1, cv2.getTrackbarPos("Kernel", "MASK"))

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        # 🔥 anti-fragmentation
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h

            if area < self.min_area:
                continue

            detections.append((int(x), int(y), int(w), int(h)))

        # ✅ ALWAYS RETURN BOTH
        return detections, mask

    def show(self, frame, blobs, intersections, counts, mask):
        vis = frame.copy()

        for b in blobs:
            cv2.rectangle(vis, (b.x, b.y), (b.x+b.w, b.y+b.h), (0,255,0), 2)

        for (x, y) in intersections:
            cv2.circle(vis, (x, y), 8, (0,0,255), -1)

        cv2.putText(vis,
            f"T:{self.threshold} A:{self.min_area}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.imshow("RADAR", vis)
        cv2.imshow("MASK", mask)

        return cv2.waitKey(1) & 0xFF == ord('q')