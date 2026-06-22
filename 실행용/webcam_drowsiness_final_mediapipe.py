import os
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image


# =========================
# 경로 설정
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EYE_TFLITE_PATH = os.path.join(BASE_DIR, "models", "eye_model.tflite")
MOUTH_TFLITE_PATH = os.path.join(BASE_DIR, "models", "mouth_model.tflite")


# =========================
# 기본 설정
# =========================

WINDOW_NAME = "Final Drowsiness Detection - MediaPipe + TFLite"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

EYE_IMAGE_SIZE = 64
MOUTH_IMAGE_SIZE = 96

INIT_STEP_TIME = 3.0

DEFAULT_EAR_THRESHOLD = 0.23
DEFAULT_MAR_THRESHOLD = 0.60

EAR_THRESHOLD_RATIO = 0.35
MAR_THRESHOLD_RATIO = 1.8

EYE_CLOSED_TIME = 0.7
YAWN_TIME = 1.0
FUSION_HOLD_TIME = 0.7

DL_INTERVAL = 3

EYE_MODEL_WEIGHT = 0.20
MOUTH_MODEL_WEIGHT = 0.15
EAR_RULE_WEIGHT = 0.45
MAR_RULE_WEIGHT = 0.20

FINAL_THRESHOLD = 0.50

EYE_CLOSED_PROB_THRESHOLD = 0.60
YAWN_PROB_THRESHOLD = 0.60

BUTTON_X1, BUTTON_Y1 = 470, 420
BUTTON_X2, BUTTON_Y2 = 630, 465

EYE_OUTPUT_IS_CLOSED_PROB = True
MOUTH_OUTPUT_IS_YAWN_PROB = False


# =========================
# MediaPipe 설정
# =========================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# Landmark index
# =========================

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
MOUTH_MAR = [61, 81, 13, 311, 308, 402, 14, 178]

LEFT_EYE_CROP = [33, 133, 160, 158, 153, 144, 159, 145]
RIGHT_EYE_CROP = [362, 263, 385, 387, 373, 380, 386, 374]
MOUTH_CROP = [61, 291, 13, 14, 78, 308, 81, 178, 402, 311]


# =========================
# TFLite 모델
# =========================

class TFLiteBinaryClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

        with open(model_path, "rb") as f:
            model_content = f.read()

        self.interpreter = tf.lite.Interpreter(model_content=model_content)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("TFLite 모델 로드 완료:", model_path)
        print("Input :", self.input_details[0]["shape"], self.input_details[0]["dtype"])
        print("Output:", self.output_details[0]["shape"], self.output_details[0]["dtype"])

    def predict(self, image):
        input_dtype = self.input_details[0]["dtype"]
        input_data = np.expand_dims(image, axis=0).astype(input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        output = np.array(output).reshape(-1)

        return float(output[0])

# =========================
# 유틸
# =========================

def put_korean_text(frame, text, position, font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(points):
    A = euclidean(points[1], points[5])
    B = euclidean(points[2], points[4])
    C = euclidean(points[0], points[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(points):
    vertical_1 = euclidean(points[1], points[7])
    vertical_2 = euclidean(points[2], points[6])
    vertical_3 = euclidean(points[3], points[5])
    horizontal = euclidean(points[0], points[4])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)


def get_points(landmarks, indices, width, height):
    points = []

    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))

    return points


def crop_by_points(frame, points, padding=12):
    h, w = frame.shape[:2]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_crop(crop, size):
    crop = apply_clahe(crop)
    crop = cv2.resize(crop, (size, size))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    return crop


# =========================
# 초기화 상태
# =========================

def reset_calibration():
    global calibration_mode, calibration_start_time
    global step1_ear_values, step1_mar_values
    global step2_ear_values, step2_mar_values
    global personal_ear_threshold, personal_mar_threshold
    global eye_closed_start, yawn_start, fusion_start
    global drowsy_state
    global eye_closed_prob, yawn_prob, fusion_score

    calibration_mode = True
    calibration_start_time = time.time()

    step1_ear_values = []
    step1_mar_values = []
    step2_ear_values = []
    step2_mar_values = []

    personal_ear_threshold = DEFAULT_EAR_THRESHOLD
    personal_mar_threshold = DEFAULT_MAR_THRESHOLD

    eye_closed_start = None
    yawn_start = None
    fusion_start = None
    drowsy_state = False

    eye_closed_prob = 0.0
    yawn_prob = 0.0
    fusion_score = 0.0

    print("초기화를 다시 시작합니다.")


def finish_calibration():
    global calibration_mode
    global personal_ear_threshold, personal_mar_threshold

    if len(step1_ear_values) == 0 or len(step2_ear_values) == 0:
        print("초기화 실패: 기본 threshold 사용")
        calibration_mode = False
        return

    open_ear = max(step1_ear_values)
    closed_ear = min(step2_ear_values)

    all_mar_values = step1_mar_values + step2_mar_values
    closed_mouth_mar = min(all_mar_values)

    if open_ear > closed_ear:
        personal_ear_threshold = closed_ear + (open_ear - closed_ear) * EAR_THRESHOLD_RATIO
    else:
        personal_ear_threshold = DEFAULT_EAR_THRESHOLD

    personal_mar_threshold = max(0.05, closed_mouth_mar * MAR_THRESHOLD_RATIO)

    calibration_mode = False

    print("초기화 완료")
    print(f"Open EAR: {open_ear:.4f}")
    print(f"Closed EAR: {closed_ear:.4f}")
    print(f"EAR TH: {personal_ear_threshold:.4f}")
    print(f"MAR TH: {personal_mar_threshold:.4f}")


def draw_calibration_ui(frame, elapsed):
    cv2.rectangle(frame, (20, 20), (760, 120), (0, 0, 0), -1)

    if elapsed < INIT_STEP_TIME:
        remain = INIT_STEP_TIME - elapsed
        message = "초기화 1단계: 눈을 뜨고 입을 닫아주세요"
        sub = f"{remain:.1f}초 후 다음 단계로 넘어갑니다"
    else:
        remain = INIT_STEP_TIME * 2 - elapsed
        message = "초기화 2단계: 눈을 감고 입을 닫아주세요"
        sub = f"{remain:.1f}초 후 초기화가 완료됩니다"

    frame = put_korean_text(frame, message, (35, 42), 28, (0, 255, 255))
    frame = put_korean_text(frame, sub, (35, 80), 23, (255, 255, 255))

    return frame


def draw_reset_button(frame):
    cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (50, 50, 50), -1)
    cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (255, 255, 255), 2)
    frame = put_korean_text(frame, "초기화", (BUTTON_X1 + 35, BUTTON_Y1 + 8), 24, (255, 255, 255))
    return frame


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if BUTTON_X1 <= x <= BUTTON_X2 and BUTTON_Y1 <= y <= BUTTON_Y2:
            reset_calibration()


# =========================
# 실행
# =========================

eye_model = TFLiteBinaryClassifier(EYE_TFLITE_PATH)
mouth_model = TFLiteBinaryClassifier(MOUTH_TFLITE_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

calibration_mode = True
calibration_start_time = time.time()

step1_ear_values = []
step1_mar_values = []
step2_ear_values = []
step2_mar_values = []

personal_ear_threshold = DEFAULT_EAR_THRESHOLD
personal_mar_threshold = DEFAULT_MAR_THRESHOLD

eye_closed_start = None
yawn_start = None
fusion_start = None

drowsy_count = 0
drowsy_state = False

eye_closed_prob = 0.0
yawn_prob = 0.0
fusion_score = 0.0

frame_count = 0

print("실행 중입니다.")
print("0~3초: 눈을 뜨고 입을 닫으세요.")
print("3~6초: 눈을 감고 입을 닫으세요.")
print("r: 초기화 다시 실행")
print("q: 종료")


while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("웹캠 프레임을 불러오지 못했습니다.")
        break

    frame_count += 1

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)

    # 저조도 보정
    frame = apply_clahe(frame)

    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "NO FACE"
    alert_message = ""

    ear = 0.0
    mar = 0.0
    ear_rule_score = 0.0
    mar_rule_score = 0.0

    current_time = time.time()

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        left_eye_ear_points = get_points(landmarks, LEFT_EYE_EAR, width, height)
        right_eye_ear_points = get_points(landmarks, RIGHT_EYE_EAR, width, height)
        mouth_mar_points = get_points(landmarks, MOUTH_MAR, width, height)

        left_eye_crop_points = get_points(landmarks, LEFT_EYE_CROP, width, height)
        right_eye_crop_points = get_points(landmarks, RIGHT_EYE_CROP, width, height)
        mouth_crop_points = get_points(landmarks, MOUTH_CROP, width, height)

        left_ear = eye_aspect_ratio(left_eye_ear_points)
        right_ear = eye_aspect_ratio(right_eye_ear_points)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth_mar_points)

        if calibration_mode:
            elapsed = current_time - calibration_start_time
            status = "CALIBRATION"

            if elapsed < INIT_STEP_TIME:
                step1_ear_values.append(ear)
                step1_mar_values.append(mar)

            elif elapsed < INIT_STEP_TIME * 2:
                step2_ear_values.append(ear)
                step2_mar_values.append(mar)

            else:
                finish_calibration()
                status = "NORMAL"

        else:
            left_crop, left_box = crop_by_points(frame, left_eye_crop_points, padding=12)
            right_crop, right_box = crop_by_points(frame, right_eye_crop_points, padding=12)
            mouth_crop, mouth_box = crop_by_points(frame, mouth_crop_points, padding=22)

            if frame_count % DL_INTERVAL == 0:
                eye_probs = []

                if left_crop is not None:
                    left_input = preprocess_crop(left_crop, EYE_IMAGE_SIZE)
                    eye_raw = eye_model.predict(left_input)

                    if EYE_OUTPUT_IS_CLOSED_PROB:
                        eye_probs.append(eye_raw)
                    else:
                        eye_probs.append(1.0 - eye_raw)

                if right_crop is not None:
                    right_input = preprocess_crop(right_crop, EYE_IMAGE_SIZE)
                    eye_raw = eye_model.predict(right_input)

                    if EYE_OUTPUT_IS_CLOSED_PROB:
                        eye_probs.append(eye_raw)
                    else:
                        eye_probs.append(1.0 - eye_raw)

                if eye_probs:
                    eye_closed_prob = float(np.mean(eye_probs))

                if mouth_crop is not None:
                    mouth_input = preprocess_crop(mouth_crop, MOUTH_IMAGE_SIZE)
                    mouth_raw = mouth_model.predict(mouth_input)

                    if MOUTH_OUTPUT_IS_YAWN_PROB:
                        yawn_prob = mouth_raw
                    else:
                        yawn_prob = 1.0 - mouth_raw

            ear_rule_score = 1.0 if ear < personal_ear_threshold else 0.0
            mar_rule_score = 1.0 if mar > personal_mar_threshold else 0.0

            fusion_score = (
                EYE_MODEL_WEIGHT * eye_closed_prob
                + MOUTH_MODEL_WEIGHT * yawn_prob
                + EAR_RULE_WEIGHT * ear_rule_score
                + MAR_RULE_WEIGHT * mar_rule_score
            )

            if ear < personal_ear_threshold or eye_closed_prob >= EYE_CLOSED_PROB_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = current_time
            else:
                eye_closed_start = None

            if mar > personal_mar_threshold or yawn_prob >= YAWN_PROB_THRESHOLD:
                if yawn_start is None:
                    yawn_start = current_time
            else:
                yawn_start = None

            if fusion_score >= FINAL_THRESHOLD and (ear_rule_score == 1.0 or mar_rule_score == 1.0):
                if fusion_start is None:
                    fusion_start = current_time
            else:
                fusion_start = None

            drowsy_detected = False

            if eye_closed_start is not None and current_time - eye_closed_start >= EYE_CLOSED_TIME:
                drowsy_detected = True
                alert_message = "Eyes closed too long"

            if yawn_start is not None and current_time - yawn_start >= YAWN_TIME:
                drowsy_detected = True
                alert_message = "Yawning detected"

            if fusion_start is not None and current_time - fusion_start >= FUSION_HOLD_TIME:
                drowsy_detected = True
                alert_message = "Fusion score detected drowsiness"

            if drowsy_detected:
                status = "DROWSINESS ALERT"

                if not drowsy_state:
                    drowsy_count += 1
                    drowsy_state = True
            else:
                status = "NORMAL"
                drowsy_state = False

            if left_crop is not None:
                cv2.rectangle(frame, left_box[:2], left_box[2:], (255, 255, 0), 1)

            if right_crop is not None:
                cv2.rectangle(frame, right_box[:2], right_box[2:], (255, 255, 0), 1)

            if mouth_crop is not None:
                cv2.rectangle(frame, mouth_box[:2], mouth_box[2:], (0, 255, 255), 1)

        for point in left_eye_ear_points + right_eye_ear_points + mouth_mar_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=1,
                circle_radius=1
            )
        )

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"EAR TH: {personal_ear_threshold:.2f}", (30, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"MAR TH: {personal_mar_threshold:.2f}", (30, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Eye Closed Prob: {eye_closed_prob:.2f}", (30, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Yawn Prob: {yawn_prob:.2f}", (30, 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Fusion Score: {fusion_score:.2f}", (30, 325),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Drowsy Count: {drowsy_count}", (30, 355),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    else:
        eye_closed_start = None
        yawn_start = None
        fusion_start = None
        drowsy_state = False

    if calibration_mode:
        elapsed = current_time - calibration_start_time
        frame = draw_calibration_ui(frame, elapsed)

    elif status == "DROWSINESS ALERT":
        cv2.putText(frame, status, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 255), 3)

        cv2.rectangle(frame, (20, 375), (620, 445), (0, 0, 255), -1)

        cv2.putText(frame, alert_message, (40, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    elif status == "NORMAL":
        cv2.putText(frame, "NORMAL", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "NO FACE DETECTED", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    frame = draw_reset_button(frame)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        reset_calibration()

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()