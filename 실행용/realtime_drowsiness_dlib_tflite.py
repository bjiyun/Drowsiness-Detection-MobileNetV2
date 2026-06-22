import time
import cv2
import dlib
import numpy as np
import tensorflow as tf

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_PATH = "haarcascade_frontalface_default.xml"

EYE_TFLITE_PATH = "models/eye_model.tflite"
MOUTH_TFLITE_PATH = "models/mouth_model.tflite"

EYE_IMAGE_SIZE = 64
MOUTH_IMAGE_SIZE = 96

EYE_CLOSED_THRESHOLD = 0.6
YAWN_THRESHOLD = 0.6

EYE_CLOSED_SECONDS = 0.7
YAWN_SECONDS = 1.0


class TFLiteBinaryClassifier:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, image):
        input_dtype = self.input_details[0]["dtype"]
        input_data = np.expand_dims(image, axis=0).astype(input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return float(output[0][0])


class DrowsinessLogic:
    def __init__(self):
        self.eye_closed_start = None
        self.yawn_start = None
        self.drowsy_state = False
        self.drowsy_count = 0

    def update(self, closed_prob, yawn_prob):
        now = time.time()

        if closed_prob >= EYE_CLOSED_THRESHOLD:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
        else:
            self.eye_closed_start = None

        if yawn_prob >= YAWN_THRESHOLD:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        drowsy = False
        reason = ""

        if self.eye_closed_start is not None and now - self.eye_closed_start >= EYE_CLOSED_SECONDS:
            drowsy = True
            reason = "Eyes closed too long"

        if self.yawn_start is not None and now - self.yawn_start >= YAWN_SECONDS:
            drowsy = True
            reason = "Yawning detected"

        if drowsy:
            if not self.drowsy_state:
                self.drowsy_count += 1
                self.drowsy_state = True
            return "DROWSINESS ALERT", reason, self.drowsy_count

        self.drowsy_state = False
        return "NORMAL", "", self.drowsy_count


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


def preprocess(crop, size):
    crop = cv2.resize(crop, (size, size))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop.astype(np.float32)


def main():
    eye_model = TFLiteBinaryClassifier(EYE_TFLITE_PATH)
    mouth_model = TFLiteBinaryClassifier(MOUTH_TFLITE_PATH)
    logic = DrowsinessLogic()

    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    window_name = "Drowsiness Detection - dlib + TFLite"

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        status = "NO FACE"
        reason = ""
        closed_prob = 0.0
        yawn_prob = 0.0
        drowsy_count = logic.drowsy_count

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(frame, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            mouth = landmarks[48:68]

            eye_probs = []

            left_crop, left_box = crop_region(frame, left_eye, padding=18)
            right_crop, right_box = crop_region(frame, right_eye, padding=18)
            mouth_crop, mouth_box = crop_region(frame, mouth, padding=25)

            if left_crop is not None:
                left_input = preprocess(left_crop, EYE_IMAGE_SIZE)
                open_prob = eye_model.predict(left_input)
                eye_probs.append(1.0 - open_prob)
                cv2.rectangle(frame, left_box[:2], left_box[2:], (255, 255, 255), 1)

            if right_crop is not None:
                right_input = preprocess(right_crop, EYE_IMAGE_SIZE)
                open_prob = eye_model.predict(right_input)
                eye_probs.append(1.0 - open_prob)
                cv2.rectangle(frame, right_box[:2], right_box[2:], (255, 255, 255), 1)

            if eye_probs:
                closed_prob = sum(eye_probs) / len(eye_probs)

            if mouth_crop is not None:
                mouth_input = preprocess(mouth_crop, MOUTH_IMAGE_SIZE)
                yawn_prob = mouth_model.predict(mouth_input)
                cv2.rectangle(frame, mouth_box[:2], mouth_box[2:], (255, 255, 255), 1)

            status, reason, drowsy_count = logic.update(closed_prob, yawn_prob)

        color = (0, 255, 0) if status == "NORMAL" else (0, 0, 255) if status == "DROWSINESS ALERT" else (0, 255, 255)

        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Eye closed prob: {closed_prob:.2f}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawn prob: {yawn_prob:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Drowsy count: {drowsy_count}", (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if reason:
            cv2.putText(frame, reason, (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()