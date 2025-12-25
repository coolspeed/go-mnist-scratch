# 코딩 에이전트용 가이드

이 문서는 코딩 에이전트(Cursor, GitHub Copilot, Claude Code 등)가 프로젝트를 이해하고 구현할 때 참고할 수 있는 상세 가이드입니다.

## 역할 및 태스크

**Role:** 숙련된 Golang 백엔드 엔지니어이자 ML 연구원

**Task:** Go언어만을 사용하여 외부 ML 라이브러리(TensorFlow, PyTorch 등) 없이, 밑바닥부터(Scratch) MNIST 손글씨 인식 프로그램을 구현

## 요구사항

### Pure Go Implementation
- 행렬 연산부터 신경망 로직까지 순수 Go 코드(`[][]float64` 슬라이스 활용)로 구현
- gonum 같은 외부 수치 연산 라이브러리 사용 지양

## 아키텍처

### 신경망 구조
- **입력층:** 784 (28x28 픽셀)
- **은닉층:** 200 (활성화 함수: Sigmoid)
- **출력층:** 10 (활성화 함수: Softmax)

### 학습 하이퍼파라미터
- **Learning Rate:** 0.3
- **Epochs:** 20
- **Batch Size:** 64

## 구현 컴포넌트

### Phase 1: matrix 패키지
`matrix/matrix.go` - 기본 행렬 연산 구현

**필요 기능:**
- `NewMatrix` - 행렬 생성
- `DotProduct` - 행렬 곱 (CPU 성능의 핵심)
- `Add`, `Subtract` - 행렬 덧셈/뺄셈
- `Apply` - 요소별 연산 (활성화 함수 적용용)
- `Transpose` - 전치 행렬

`matrix/matrix_test.go` - 유닛 테스트
- 각 연산의 정확성을 보장하는 테스트 코드 필수

### Phase 2: utils 패키지
`utils/loader.go` - MNIST 데이터 로더

MNIST는 IDX 바이너리 포맷을 사용합니다. `encoding/binary` 패키지를 사용하여 파싱:

- 헤더 파싱 (Magic number, 개수, 행, 열)
- 이미지 데이터를 `[]float64` (0.0~1.0 정규화)로 변환
- 레이블 데이터를 One-hot encoding 벡터로 변환

`setup.sh` - 데이터 다운로드 자동화
- `curl`을 사용하여 Yann LeCun 교수님의 웹사이트나 미러 사이트에서 데이터셋 다운로드

### Phase 3: neural 패키지
`neural/network.go` - 신경망 모델

**Network 구조체:**
- `Train()` 메서드 - 역전파(Backpropagation) 포함
- `Predict()` 메서드 - 순전파(Forward Propagation)

**수학적 로직:**
- **활성화 함수:**
  - Hidden Layer: Sigmoid
  - Output Layer: Softmax (다중 분류 확률 계산)
- **손실 함수:** Cross-Entropy Loss
- **가중치 초기화:** Xavier 또는 He initialization 권장
- **순전파:** Input * W1 -> Sigmoid -> * W2 -> Softmax -> Output
- **역전파:** 오차 역전파를 통한 가중치 업데이트 (Gradient Descent)

`neural/activations.go` - 활성화 함수 구현
- Sigmoid, ReLU, Softmax 등

### Phase 4: Model Persistence
학습된 가중치(Weights)와 편향(Biases)을 저장하고 로드하는 기능 필수:

- `SaveModel(filename string)` - JSON 또는 Gob 형식으로 저장
- `LoadModel(filename string)` - 저장된 모델 로드

### Phase 5: 애플리케이션

`main.go` - 학습 실행 CLI
- 데이터 로드
- 에폭(Epoch) 반복 학습
- 학습된 모델을 파일로 저장

`server.go` - 추론 웹 서버
- 서버 시작 시 미리 저장된 모델 파일 로드
- `net/http`를 사용해 REST API 구축 (`POST /predict`)
- HTML/JS 프론트엔드: 사용자가 마우스로 숫자를 그리면 28x28 배열로 변환해 서버로 전송
- 서버는 결과를 JSON으로 응답

### Phase 6: 최적화 (Bonus)

**도전 과제:**
- **Goroutine 병렬화:** `DotProduct` 연산이나 배치 학습 시 Goroutine을 활용하여 연산 속도 가속
- **메모리 최적화:** 불필요한 슬라이스 할당을 줄이고 메모리 재사용 기법 적용

## 성능 목표

- **추론 속도:** 단일 예측 < 10ms (CPU 기준)
- **정확도:** 테스트셋 기준 90% 이상

## 코딩 가이드라인

- 모듈화가 잘 되어야 함
- 각 함수에는 명확한 주석 작성
- CPU에서 매우 빠르고 가볍게 동작하는 것이 핵심 목표

## 참고 자료

- IDX 파일 형식: http://yann.lecun.com/exdb/mnist/
