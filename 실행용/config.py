# config.py
# 공통 설정 파일

from pathlib import Path

# =========================
# 1. 경로 설정
# =========================
# 한글 경로 문제를 피하기 위해 datasets 폴더만 영어 경로로 이동한 상태 기준
DATASET_ROOT = Path(r"D:\yonsei\2026\4-1\datasets")

EYE_DATA_DIR = DATASET_ROOT / "eye"
MOUTH_DATA_DIR = DATASET_ROOT / "mouth"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

EYE_MODEL_PATH = str(MODEL_DIR / "eye_model.h5")
MOUTH_MODEL_PATH = str(MODEL_DIR / "mouth_model.h5")

EYE_TFLITE_PATH = str(MODEL_DIR / "eye_model.tflite")
MOUTH_TFLITE_PATH = str(MODEL_DIR / "mouth_model.tflite")

# 기존 dlib/EAR-MAR 방식에서 사용하는 파일
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_PATH = "haarcascade_frontalface_default.xml"

# =========================
# 2. 입력 크기
# =========================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

EYE_IMAGE_SIZE = 64
MOUTH_IMAGE_SIZE = 96

# =========================
# 3. 기존 dlib/EAR-MAR 방식 설정
# =========================
FACE_DETECT_INTERVAL = 5
ROI_MARGIN = 40

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.70

# =========================
# 4. 딥러닝 추론 threshold
# =========================
# eye 모델의 폴더 순서가 ['closed', 'open']이면 sigmoid 출력은 open 확률입니다.
# realtime 코드에서 closed_prob = 1 - open_prob로 변환합니다.
EYE_CLOSED_THRESHOLD = 0.60

# mouth 모델의 폴더 순서가 ['normal', 'yawn']이면 sigmoid 출력은 yawn 확률입니다.
YAWN_THRESHOLD = 0.60

EYE_CLOSED_SECONDS = 0.7
YAWN_SECONDS = 1.0

# =========================
# 5. 학습 설정
# =========================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
SEED = 42

# =========================
# 6. 화면 표시 여부
# =========================
DRAW_LANDMARKS = True
DRAW_DEBUG_TEXT = True
