# 지능형멀티미디어시스템-2026
# 변지윤*, 이효진
# 최종 수정일: 2026.04.29

import cv2
import dlib   # 얼굴 landmark(특징점) 추출 라이브러리
import numpy as np
from scipy.spatial import distance as dist  # 두 점 사이 거리 계산 함수

PREDICTOR_PATH = r"C:\project\shape_predictor_68_face_landmarks.dat"  # dlib에서 사용하는 얼굴 68개 특징점 모델 파일
HAAR_PATH = r"C:\project\haarcascade_frontalface_default.xml"   # OpenCV에서 사용하는 얼굴 검출 모델 (Haar Cascade)

# [10] 논문 기반 Threshold 설정!!
EAR_THRESHOLD = 0.25  # 눈이 이 값보다 작으면 "눈 감음"으로 판단
EAR_CONSEC_FRAMES = 10  # 눈 감김 상태가 몇 프레임 이상 유지되면 졸음으로 판단할지

MAR_THRESHOLD = 0.70   # 입이 이 값보다 크면 "하품"으로 판단
MAR_CONSEC_FRAMES = 10  # 하품 상태가 몇 프레임 이상 유지되면 졸음으로 판단

predictor = dlib.shape_predictor(PREDICTOR_PATH)  # 얼굴 landmark 추출 모델 로드

face_cascade = cv2.CascadeClassifier(HAAR_PATH)  # Haar Cascade 얼굴 검출기 로드

if face_cascade.empty():
    print("Haar Cascade 파일을 불러오지 못했습니다.")
    exit()

# 캠 키기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

eye_counter = 0   # 눈 감김 프레임 수 카운트
mouth_counter = 0 # 하품 프레임 수 카운트
drowsy_count = 0  # 졸음 발생 횟수 누적
drowsy_state = False  # 현재 졸음 상태 여부 (중복 카운트 방지)

# EAR 계산
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # 눈 세로 거리 1
    B = dist.euclidean(eye[2], eye[4])  # 눈 세로 거리 2
    C = dist.euclidean(eye[0], eye[3])  # 눈 가로 거리
    return (A + B) / (2.0 * C)     # 눈 비율 계산

# MAR 계산 
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 입 세로 거리 1
    B = dist.euclidean(mouth[3], mouth[9])   # 입 세로 거리 2
    C = dist.euclidean(mouth[4], mouth[8])   # 입 세로 거리 3
    D = dist.euclidean(mouth[0], mouth[6])   # 입 가로 거리
    return (A + B + C) / (2.0 * D)    # 입 비율 계산

# 메인 처리
while True:
    ret, frame = cap.read()   # 웹캠에서 프레임 읽기

    if not ret or frame is None:
        print("웹캠을 불러오지 못했습니다.")
        break

    frame = cv2.flip(frame, 1)    # 좌우 반전 (거울처럼 보이게)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # 흑백 이미지
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # RGB 이미지

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,  # 이미지 축소 비율
        minNeighbors=5,   # 검출 민감도
        minSize=(80, 80)  # 최소 얼굴 크기
    )

    status = "NO FACE"
    alert_message = ""   # 경고 메시지

    # 얼굴이 있을 경우 처리
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)   # dlib이 요구하는 메모리 형태로 변환
        shape = predictor(rgb, rect)   # 얼굴 landmark 추출
        shape = np.array([[p.x, p.y] for p in shape.parts()])    # landmark를 numpy 배열로 변환

        # 눈과 입 영역 분리
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        mouth = shape[48:68]

        # EAR 계산
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # MAR 계산
        mar = mouth_aspect_ratio(mouth)

        # 눈 감김 판단
        if ear < EAR_THRESHOLD:
            eye_counter = eye_counter + 1
        else:
            eye_counter = 0

        # 하품 판단
        if mar > MAR_THRESHOLD:
            mouth_counter = mouth_counter + 1
        else:
            mouth_counter = 0

        drowsy_detected = False

        # 졸음 조건
        if eye_counter >= EAR_CONSEC_FRAMES:
            drowsy_detected = True
            alert_message = "Eyes closed too long"

        elif mouth_counter >= MAR_CONSEC_FRAMES:
            drowsy_detected = True
            alert_message = "Yawning detected"

        # 상태 업데이트
        if drowsy_detected:
            status = "DROWSINESS ALERT"

            if not drowsy_state:
                drowsy_count = drowsy_count + 1
                drowsy_state = True
        else:
            status = "NORMAL"
            drowsy_state = False

        # 얼굴 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # landmark 점 찍기
        for (px, py) in shape:
            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

        # 텍스트 출력
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Drowsy Count: {drowsy_count}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        break

    if status == "DROWSINESS ALERT":
        cv2.putText(frame, status, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        cv2.rectangle(frame, (20, 210), (620, 280), (0, 0, 255), -1)

        cv2.putText(frame, alert_message, (40, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    elif status == "NORMAL":
        cv2.putText(frame, "NORMAL", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "NO FACE DETECTED", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()