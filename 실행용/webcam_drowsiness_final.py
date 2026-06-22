# webcam_drowsiness_final.py
# dlib + EAR/MAR 베이스라인 + TFLite Eye/Mouth 모델 Fusion 최종 실행 코드

import os
import time
import cv2
import dlib
import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist


# =========================
# 경로 설정
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
HAAR_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

EYE_TFLITE_PATH = os.path.join(BASE_DIR, "models", "eye_model.tflite")
MOUTH_TFLITE_PATH = os.path.join(BASE_DIR, "models", "mouth_model.tflite")


# =========================
# 기본 설정
# =========================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

EYE_IMAGE_SIZE = 64
MOUTH_IMAGE_SIZE = 96

# 기존 EAR/MAR baseline threshold
DEFAULT_EAR_THRESHOLD = 0.25
DEFAULT_MAR_THRESHOLD = 0.70

# 사용자 초기화 시간
CALIBRATION_STEP_TIME = 3.0

# 딥러닝 확률 threshold
EYE_CLOSED_PROB_THRESHOLD = 0.60
YAWN_PROB_THRESHOLD = 0.60

# 시간 조건
EYE_CLOSED_SECONDS = 0.7
YAWN_SECONDS = 1.0

# Fusion 가중치
EYE_MODEL_WEIGHT = 0.35
MOUTH_MODEL_WEIGHT = 0.35
EAR_RULE_WEIGHT = 0.20
MAR_RULE_WEIGHT = 0.10

FINAL_THRESHOLD = 0.50

# 모바일/실시간 경량화: TFLite 추론 주기
DL_INFERENCE_INTERVAL = 3

WINDOW_NAME = "Drowsiness Detection Final"


# =========================
# TFLite 모델 클래스
# =========================

class TFLiteBinaryClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("TFLite 모델 로드 완료:", model_path)
        print("Input:", self.input_details[0]["shape"], self.input_details[0]["dtype"])
        print("Output:", self.output_details[0]["shape"], self.output_details[0]["dtype"])

    def predict(self, image):
        input_dtype = self.input_details[0]["dtype"]
        input_scale, input_zero_point = self.input_details[0].get("quantization", (0.0, 0))

        input_data = np.expand_dims(image, axis=0)

        if input_dtype == np.uint8:
            if input_scale > 0:
                input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        output = np.array(output).reshape(-1)

        if output.size == 1:
            return float(output[0])

        # 2-class 출력이면 두 번째 클래스를 positive로 사용
        exp = np.exp(output - np.max(output))
        prob = exp / np.sum(exp)
        return float(prob[1])


# =========================
# 비율 계산 함수
# =========================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[3], mouth[9])
    C = dist.euclidean(mouth[4], mouth[8])
    D = dist.euclidean(mouth[0], mouth[6])

    if D == 0:
        return 0.0

    return (A + B + C) / (2.0 * D)


# =========================
# 이미지 전처리
# =========================

def enhance_low_light(image):
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def crop_region(frame, points, padding=15):
    h, w = frame.shape[:2]

    xs = points[:, 0]
    ys = points[:, 1]

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + padding)
    y2 = min(h, int(ys.max()) + padding)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_crop(crop, size):
    crop = enhance_low_light(crop)
    crop = cv2.resize(crop, (size, size))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    return crop


# =========================
# 사용자 초기화
# =========================

class CalibrationManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = True
        self.start_time = time.time()

        self.step1_ears = []
        self.step1_mars = []
        self.step2_ears = []
        self.step2_mars = []

        self.ear_threshold = DEFAULT_EAR_THRESHOLD
        self.mar_threshold = DEFAULT_MAR_THRESHOLD

        print("초기화 시작")
        print("0~3초: 눈을 뜨고 입을 닫으세요.")
        print("3~6초: 눈을 감고 입을 닫으세요.")

    def update(self, ear, mar):
        if not self.active:
            return

        elapsed = time.time() - self.start_time

        if elapsed < CALIBRATION_STEP_TIME:
            self.step1_ears.append(ear)
            self.step1_mars.append(mar)

        elif elapsed < CALIBRATION_STEP_TIME * 2:
            self.step2_ears.append(ear)
            self.step2_mars.append(mar)

        else:
            self.finish()

    def finish(self):
        if len(self.step1_ears) == 0 or len(self.step2_ears) == 0:
            print("초기화 실패: 기본 threshold 사용")
            self.active = False
            return

        open_ear = max(self.step1_ears)
        closed_ear = min(self.step2_ears)

        closed_mouth_mar = min(self.step1_mars + self.step2_mars)

        if open_ear > closed_ear:
            self.ear_threshold = closed_ear + (open_ear - closed_ear) * 0.35
        else:
            self.ear_threshold = DEFAULT_EAR_THRESHOLD

        self.mar_threshold = max(0.05, closed_mouth_mar * 1.8)

        self.active = False

        print("초기화 완료")
        print(f"Open EAR: {open_ear:.4f}")
        print(f"Closed EAR: {closed_ear:.4f}")
        print(f"EAR TH: {self.ear_threshold:.4f}")
        print(f"MAR TH: {self.mar_threshold:.4f}")

    def draw(self, frame):
        if not self.active:
            return frame

        elapsed = time.time() - self.start_time

        cv2.rectangle(frame, (20, 20), (760, 100), (0, 0, 0), -1)

        if elapsed < CALIBRATION_STEP_TIME:
            msg = "CALIBRATION 1: Open eyes, close mouth"
            sub = f"{CALIBRATION_STEP_TIME - elapsed:.1f}s remaining"
        else:
            msg = "CALIBRATION 2: Close eyes, close mouth"
            sub = f"{CALIBRATION_STEP_TIME * 2 - elapsed:.1f}s remaining"

        cv2.putText(frame, msg, (35, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, sub, (35, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return frame


# =========================
# 졸음 판단 로직
# =========================

class FusionDrowsinessLogic:
    def __init__(self):
        self.eye_closed_start = None
        self.yawn_start = None
        self.fusion_start = None

        self.drowsy_state = False
        self.drowsy_count = 0

    def reset_timers(self):
        self.eye_closed_start = None
        self.yawn_start = None
        self.fusion_start = None
        self.drowsy_state = False

    def update(
        self,
        ear,
        mar,
        closed_prob,
        yawn_prob,
        ear_threshold,
        mar_threshold
    ):
        now = time.time()

        ear_rule = 1.0 if ear < ear_threshold else 0.0
        mar_rule = 1.0 if mar > mar_threshold else 0.0

        final_score = (
            EYE_MODEL_WEIGHT * closed_prob
            + MOUTH_MODEL_WEIGHT * yawn_prob
            + EAR_RULE_WEIGHT * ear_rule
            + MAR_RULE_WEIGHT * mar_rule
        )

        if ear < ear_threshold or closed_prob >= EYE_CLOSED_PROB_THRESHOLD:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
        else:
            self.eye_closed_start = None

        if mar > mar_threshold or yawn_prob >= YAWN_PROB_THRESHOLD:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        if final_score >= FINAL_THRESHOLD:
            if self.fusion_start is None:
                self.fusion_start = now
        else:
            self.fusion_start = None

        drowsy = False
        reason = ""

        if self.eye_closed_start is not None and now - self.eye_closed_start >= EYE_CLOSED_SECONDS:
            drowsy = True
            reason = "Eyes closed too long"

        if self.yawn_start is not None and now - self.yawn_start >= YAWN_SECONDS:
            drowsy = True
            reason = "Yawning detected"

        if self.fusion_start is not None and now - self.fusion_start >= 0.7:
            drowsy = True
            reason = "Fusion score detected drowsiness"

        if drowsy:
            status = "DROWSINESS ALERT"
            if not self.drowsy_state:
                self.drowsy_count += 1
                self.drowsy_state = True
        else:
            status = "NORMAL"
            self.drowsy_state = False

        return status, reason, self.drowsy_count, final_score, ear_rule, mar_rule


# =========================
# 메인
# =========================

def main():
    print("실행 파일 기준 경로:", BASE_DIR)

    eye_model = TFLiteBinaryClassifier(EYE_TFLITE_PATH)
    mouth_model = TFLiteBinaryClassifier(MOUTH_TFLITE_PATH)

    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)

    if face_cascade.empty():
        print("Haar Cascade 파일을 불러오지 못했습니다.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    calibration = CalibrationManager()
    logic = FusionDrowsinessLogic()

    frame_count = 0

    closed_prob = 0.0
    yawn_prob = 0.0
    final_score = 0.0

    print("실행 중입니다.")
    print("q: 종료")
    print("r: 초기화 다시 실행")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("웹캠 프레임을 불러오지 못했습니다.")
            break

        frame_count += 1

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        status = "NO FACE"
        reason = ""

        ear = 0.0
        mar = 0.0
        drowsy_count = logic.drowsy_count

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(np.ascontiguousarray(rgb, dtype=np.uint8), rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            mouth = landmarks[48:68]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            calibration.update(ear, mar)

            left_crop, left_box = crop_region(frame, left_eye, padding=18)
            right_crop, right_box = crop_region(frame, right_eye, padding=18)
            mouth_crop, mouth_box = crop_region(frame, mouth, padding=25)

            if not calibration.active and frame_count % DL_INFERENCE_INTERVAL == 0:
                eye_probs = []

                if left_crop is not None:
                    left_input = preprocess_crop(left_crop, EYE_IMAGE_SIZE)
                    open_prob = eye_model.predict(left_input)
                    eye_probs.append(1.0 - open_prob)

                if right_crop is not None:
                    right_input = preprocess_crop(right_crop, EYE_IMAGE_SIZE)
                    open_prob = eye_model.predict(right_input)
                    eye_probs.append(1.0 - open_prob)

                if eye_probs:
                    closed_prob = float(np.mean(eye_probs))

                if mouth_crop is not None:
                    mouth_input = preprocess_crop(mouth_crop, MOUTH_IMAGE_SIZE)
                    yawn_prob = mouth_model.predict(mouth_input)

            if calibration.active:
                status = "CALIBRATION"
                logic.reset_timers()
            else:
                status, reason, drowsy_count, final_score, ear_rule, mar_rule = logic.update(
                    ear=ear,
                    mar=mar,
                    closed_prob=closed_prob,
                    yawn_prob=yawn_prob,
                    ear_threshold=calibration.ear_threshold,
                    mar_threshold=calibration.mar_threshold
                )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            for px, py in landmarks:
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            if left_box is not None:
                cv2.rectangle(frame, left_box[:2], left_box[2:], (255, 255, 0), 1)

            if right_box is not None:
                cv2.rectangle(frame, right_box[:2], right_box[2:], (255, 255, 0), 1)

            if mouth_box is not None:
                cv2.rectangle(frame, mouth_box[:2], mouth_box[2:], (0, 255, 255), 1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(frame, f"EAR TH: {calibration.ear_threshold:.2f}", (30, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(frame, f"MAR TH: {calibration.mar_threshold:.2f}", (30, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(frame, f"Eye closed prob: {closed_prob:.2f}", (30, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(frame, f"Yawn prob: {yawn_prob:.2f}", (30, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(frame, f"Fusion score: {final_score:.2f}", (30, 295),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(frame, f"Drowsy count: {drowsy_count}", (30, 325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        else:
            logic.reset_timers()

        if status == "CALIBRATION":
            frame = calibration.draw(frame)

        elif status == "DROWSINESS ALERT":
            cv2.putText(frame, status, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 255), 3)

            cv2.rectangle(frame, (20, 350), (620, 420), (0, 0, 255), -1)

            cv2.putText(frame, reason, (40, 395),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif status == "NORMAL":
            cv2.putText(frame, "NORMAL", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        else:
            cv2.putText(frame, "NO FACE DETECTED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            calibration.reset()
            logic.reset_timers()
            closed_prob = 0.0
            yawn_prob = 0.0
            final_score = 0.0

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()