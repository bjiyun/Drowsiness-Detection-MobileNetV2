# drowsiness_logic.py
# 기존 dlib + EAR/MAR 방식에서 사용하는 시간 기반 졸음 판단 로직

import time


class DrowsinessDetector:
    def __init__(
        self,
        ear_threshold,
        mar_threshold,
        eye_closed_seconds,
        yawn_seconds
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.eye_closed_seconds = eye_closed_seconds
        self.yawn_seconds = yawn_seconds

        self.eye_closed_start = None
        self.yawn_start = None

        self.drowsy_state = False
        self.drowsy_count = 0

    def update(self, ear, mar):
        now = time.time()

        alert_message = ""
        drowsy_detected = False

        if ear < self.ear_threshold:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
        else:
            self.eye_closed_start = None

        if mar > self.mar_threshold:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        if self.eye_closed_start is not None:
            if now - self.eye_closed_start >= self.eye_closed_seconds:
                drowsy_detected = True
                alert_message = "Eyes closed too long"

        if self.yawn_start is not None:
            if now - self.yawn_start >= self.yawn_seconds:
                drowsy_detected = True
                alert_message = "Yawning detected"

        if drowsy_detected:
            status = "DROWSINESS ALERT"

            if not self.drowsy_state:
                self.drowsy_count += 1
                self.drowsy_state = True
        else:
            status = "NORMAL"
            self.drowsy_state = False

        return status, alert_message, self.drowsy_count
