# 졸음운전 방지를 위한 실시간 졸음 감지 시스템

MediaPipe Face Mesh와 MobileNetV2 기반 TensorFlow Lite 모델을 활용하여 운전자의 졸음 상태를 실시간으로 감지하는 시스템입니다.

눈(Eye)과 입(Mouth) 영역을 독립적으로 분석하고, EAR(Eye Aspect Ratio), MAR(Mouth Aspect Ratio) 기반 규칙과 딥러닝 예측 결과를 융합(Fusion)하여 졸음 여부를 판단합니다.

본 프로젝트는 모바일 환경에서 동작 가능한 경량화와 실시간성을 목표로 개발되었습니다.

---

# 프로젝트 개요

기존 졸음 감지 시스템은 주로 눈 감김 시간만을 이용하여 졸음을 판단합니다.

하지만 실제 운전 환경에서는 다음과 같은 문제가 발생합니다.

* 사용자별 눈 크기 차이
* 안경 착용
* 조명 변화
* 얼굴 각도 변화
* FPS 변화에 따른 오탐지

본 프로젝트는 이러한 문제를 해결하기 위해

* MediaPipe Face Mesh
* MobileNetV2 기반 CNN
* EAR / MAR 분석
* 개인 맞춤형 Calibration
* Fusion 기반 의사결정

을 결합한 졸음 감지 시스템을 구현하였습니다.

---

# 주요 기능

## Eye CNN 기반 눈 상태 분석

눈 영역(ROI)을 추출하여 TensorFlow Lite 모델로 추론합니다.

판단 항목

* Open Eye
* Closed Eye

사용 모델

```text
eye_model.tflite
```

---

## Mouth CNN 기반 하품 감지

입 영역(ROI)을 추출하여 TensorFlow Lite 모델로 추론합니다.

판단 항목

* Normal
* Yawn

사용 모델

```text
mouth_model.tflite
```

---

## EAR(Eye Aspect Ratio)

눈 랜드마크를 이용하여 눈 감김 정도를 계산합니다.

특정 임계값 이하 상태가 일정 시간 이상 유지되면 졸음 상태로 판단합니다.

---

## MAR(Mouth Aspect Ratio)

입 랜드마크를 이용하여 하품 여부를 계산합니다.

특정 임계값 이상 상태가 일정 시간 이상 유지되면 하품 상태로 판단합니다.

---

## 개인 맞춤형 Calibration

프로그램 시작 시 사용자별 기준값을 자동 생성합니다.

### 1단계

눈 뜸 + 입 닫음

### 2단계

눈 감음 + 입 닫음

위 과정을 통해

* EAR Threshold
* MAR Threshold

를 자동 계산합니다.

사용자별 얼굴 특성을 반영하여 정확도를 향상시킵니다.

---

# 시스템 구조

```text
웹캠 입력
    ↓
MediaPipe Face Mesh
    ↓
얼굴 랜드마크 추출
    ↓
ROI 추출
 ├─ Eye ROI
 └─ Mouth ROI
    ↓
TensorFlow Lite 추론
 ├─ Eye CNN
 └─ Mouth CNN
    ↓
EAR 계산
MAR 계산
    ↓
Fusion Score 계산
    ↓
졸음 여부 판단
    ↓
경고 출력
```

---

# Fusion 기반 판단

최종 졸음 점수는 다음 정보를 결합하여 계산됩니다.

* Eye CNN 결과
* Mouth CNN 결과
* EAR 분석 결과
* MAR 분석 결과

단일 방식보다 오탐지(False Positive)를 줄이고 안정성을 향상시켰습니다.

---

# 모바일 최적화

실시간 처리를 위해 다음 최적화를 적용하였습니다.

### ROI 기반 추론

전체 화면이 아닌 눈과 입 영역만 처리

### TensorFlow Lite 사용

경량 추론 모델 적용

### CLAHE 적용

저조도 환경 대응

### 단일 얼굴 추적

가장 가까운 한 명만 추적

### 시간 기반 판단

프레임 수가 아닌 시간 기준 사용

예시

* 눈 감김 0.7초 이상
* 하품 1.0초 이상

FPS 변화에 영향을 받지 않습니다.

---

# 프로젝트 구조

```text
Drowsiness-Detection-MobileNetV2
│
├── DrowsinessApp2/
│
├── models/
│   ├── eye_model.tflite
│   └── mouth_model.tflite
│
├── webcam_drowsiness_final_mediapipe_light.py
│
├── .gitignore
└── README.md
```

---

# 실험 환경

## Hardware

| Component | Specification          |
| --------- | ---------------------- |
| CPU       | Intel Core i7-1260P    |
| RAM       | 32 GB                  |
| GPU       | Intel Iris Xe Graphics |

## Software

| Component | Version           |
| --------- | ----------------- |
| OS        | Windows 11 64-bit |
| Python    | 3.11.9            |

## Libraries

| Library    | Version |
| ---------- | ------- |
| TensorFlow | 2.16.1  |
| MediaPipe  | 0.10.14 |
| OpenCV     | 4.9.0   |
| NumPy      | 1.26.4  |
| Pillow     | 12.2.0  |

---

# 설치 방법

가상환경 생성

```bash
python -m venv final_env
```

가상환경 활성화

### Windows

```bash
final_env\Scripts\activate
```

라이브러리 설치

```bash
pip install tensorflow==2.16.1
pip install mediapipe==0.10.14
pip install opencv-python==4.9.0.80
pip install numpy==1.26.4
pip install pillow==12.2.0
```

---

# 실행 방법

```bash
python webcam_drowsiness_final_mediapipe_light.py
```

---

# 사용 기술

### Computer Vision

* OpenCV
* MediaPipe Face Mesh

### Deep Learning

* TensorFlow
* TensorFlow Lite
* MobileNetV2

### Programming Language

* Python

---

# 연구 목적

본 프로젝트는 지능형멀티미디어시스템 팀 프로젝트의 일환으로 수행되었으며, 실제 운전 환경에서 활용 가능한 경량 실시간 졸음 감지 시스템 개발을 목표로 하였습니다.

---

# Authors

### 변지윤

* 데이터 전처리
* 모델 개선
* 시스템 통합
* 성능 최적화

### 이효진

* 데이터 수집
* 베이스라인 구현
* Android 애플리케이션 개발

---

# License

This project is intended for educational and research purposes only.
