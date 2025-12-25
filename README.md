# Project Plan: Go-MNIST-Scratch

**목표**: 외부 ML 라이브러리 없이, 순수 Go만 사용하여 3층 신경망(Input-Hidden-Output)을 구현하고 MNIST 데이터를 학습 및 추론한다.

## 1. 프로젝트 구조 (Project Structure)
코딩 에이전트가 헤매지 않도록 디렉토리 구조를 잡아줍니다.

```plaintext
go-mnist-scratch/
├── data/               # MNIST 데이터셋 파일 (idx1-ubyte, idx3-ubyte)
├── matrix/             # 행렬 연산 라이브러리 (직접 구현)
│   ├── matrix.go
│   └── matrix_test.go  # [NEW] 행렬 연산 유닛 테스트
├── neural/             # 신경망 모델 정의 및 학습/추론 로직
│   ├── network.go
│   └── activations.go
├── utils/              # MNIST 파일 파싱 및 이미지 처리 유틸리티
│   └── loader.go
├── main.go             # 학습 실행 및 모델 저장 (Training Mode)
├── server.go           # 웹 서버 및 추론 API (Inference Mode)
└── setup.sh            # [NEW] 데이터 다운로드설정 스크립트
```

> **참고:** IDX 파일 형식은 MNIST 데이터셋의 전용 바이너리 포맷입니다. 자세한 사양은 [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)에서 확인할 수 있습니다.

## 2. 단계별 구현 명세 (Implementation Steps)

### Phase 1: 행렬 연산 패키지 구현 (matrix 패키지)
외부 라이브러리(gonum 등)를 쓰면 편하지만, 공부 목적에 맞게 기본적인 행렬 연산을 슬라이스(`[][]float64`)로 직접 구현하도록 지시합니다.

**필요 기능:**
- 행렬 생성 (`NewMatrix`)
- 행렬 곱 (`DotProduct`) - 가장 중요 (CPU 성능의 핵심)
- 행렬 덧셈/뺄셈 (`Add`, `Subtract`)
- 요소별 연산 (`Apply` - 활성화 함수 적용용)
- 전치 행렬 (`Transpose`)
- **Unit Test (`matrix_test.go`):** 각 연산이 정확한지 검증하는 테스트 코드 작성 필수.

### Phase 2: MNIST 데이터 로더 (utils 패키지)
MNIST는 특이한 바이너리 포맷(IDX format)을 사용합니다. 이를 Go의 구조체로 변환해야 합니다.

**구현 내용:**
- `setup.sh`: `curl`을 사용하여 Yann LeCun 교수님의 웹사이트나 미러 사이트에서 데이터셋 자동 다운로드.
- `encoding/binary` 패키지를 사용하여 헤더(Magic number, 개수, 행, 열) 파싱.
- 이미지 데이터를 `[]float64` (0.0~1.0 정규화)로 변환.
- 레이블 데이터를 One-hot encoding 벡터로 변환.

### Phase 3: 신경망 모델링 (neural 패키지)
가볍고 빠른 추론을 위해 가장 기본적인 MLP(Multi-Layer Perceptron) 구조를 채택합니다.

**아키텍처:**
- **Input Layer:** 784 노드 (28x28 픽셀)
- **Hidden Layer:** 200 노드 (CPU 부하를 고려한 크기)
- **Output Layer:** 10 노드 (숫자 0~9)

**수학적 로직:**
- **활성화 함수:**
  - Hidden Layer: Sigmoid 또는 ReLU.
  - Output Layer: **Softmax** (다중 분류 확률 계산).
- **손실 함수:** **Cross-Entropy Loss** (Softmax와 결합 시 학습 효율 좋음).
- **가중치 초기화:** 랜덤 초기화 (Xavier/He initialization 권장).
- **Forward Propagation (추론용):** Input * W1 -> Sigmoid -> * W2 -> Sigmoid -> Output
- **Backpropagation (학습용):** 오차 역전파를 통한 가중치 업데이트 (Gradient Descent).

### Phase 4: 학습 및 모델 저장 (main.go)
에이전트에게 학습 루프를 작성하게 합니다.

**학습 파라미터 (권장값):**
- **Learning Rate:** 0.1 ~ 0.5 (Sigmoid 활성화 함수에 적합)
- **Epochs:** 10 ~ 30 (정확도 90%+ 달성 목표)
- **Batch Size:** 32 ~ 128 (메모리 효율성 고려)

**기능:**
- 데이터 로드.
- 에폭(Epoch) 반복.
- 학습된 가중치(Weights)와 편향(Biases)을 JSON 또는 Gob 파일로 저장. (이것이 핵심입니다. 추론 시에는 학습 없이 이 파일만 로드하면 됨).

### Phase 5: 추론 웹 서버 (server.go)
"작고 빠른 모델"의 결과를 눈으로 확인하는 단계입니다.

**성능 목표:**
- **추론 속도:** 단일 예측 < 10ms (CPU 기준)
- **정확도:** 테스트셋 기준 90% 이상

**기능:**
- 서버 시작 시 미리 저장된 모델 파일(JSON/Gob) 로드.
- `net/http`를 사용해 간단한 REST API 구축 (`POST /predict`).
- HTML/JS 프론트엔드: 사용자가 마우스로 숫자를 그리면 28x28 배열로 변환해 서버로 전송.
- 서버는 결과를 JSON으로 응답.

### Phase 6: 심화 최적화 (Optimization - Bonus)
Go 언어의 장점을 살려 성능을 극한으로 끌어올려 봅니다.

**도전 과제:**
- **Goroutine 병렬화:** `DosProduct` 연산이나 배치 학습(`Batch Training`) 시 Goroutine을 활용하여 연산 속도 가속.
- **메모리 최적화:** 불필요한 슬라이스 할당을 줄이고 메모리 재사용 기법 적용.

## 3. 코딩 에이전트용 프롬프트 (Copy & Paste)
이 플랜을 바탕으로 코딩 에이전트(예: Cursor, Github Copilot 등)에게 작업을 시킬 때 사용할 수 있는 구체적인 프롬프트입니다. 아래 내용을 복사해서 사용하세요.

---

**Role:** 너는 숙련된 Golang 백엔드 엔지니어이자 ML 연구원이야.

**Task:** Go언어만을 사용하여 외부 무거운 ML 라이브러리(TensorFlow, PyTorch 등) 없이, 밑바닥부터(Scratch) 구현하는 MNIST 손글씨 인식 프로그램을 작성해줘.

**Requirements:**
- **Pure Go Implementation:** 행렬 연산부터 신경망 로직까지 순수 Go 코드(`[][]float64` 슬라이스 활용)로 구현할 것. (gonum 사용 지양)

**Architecture:**
- **입력층:** 784 (28x28)
- **은닉층:** 200 (활성화 함수: Sigmoid)
- **출력층:** 10 (활성화 함수: Sigmoid)

**Training Hyperparameters:**
- **Learning Rate:** 0.3
- **Epochs:** 20
- **Batch Size:** 64

**Components:**
- `matrix/matrix_test.go`: 행렬 연산의 정확성을 보장하는 유닛 테스트 필수 포함.
- `utils/loader.go`: MNIST IDX 바이너리 파일을 읽어서 파싱하는 로직. `setup.sh` 스크립트로 데이터 다운로드 자동화.
- `neural/network.go`: `Train()`(역전파 포함)과 `Predict()`(순전파) 메서드를 가진 Network 구조체. (Output Layer는 Softmax, Loss는 Cross-Entropy 권장)
- **Model Persistence:** 학습된 가중치(Weights)를 JSON 파일로 저장하고, 다시 불러올 수 있는 기능 필수.

**Application:**
- `main.go`: 데이터를 학습시키고 모델을 파일로 저장하는 CLI 툴.
- `server.go`: 저장된 모델을 로드하여, HTTP 요청으로 들어온 이미지 데이터를 추론하는 초경량 웹 서버.

**Optimization (Bonus):**
- 행렬 연산이나 배치 처리에 **Goroutine**을 사용하여 병렬 처리 성능 최적화 도전.

**Goal:** CPU에서 매우 빠르고 가볍게 돌아가는 것이 핵심 목표임. 코드는 모듈화가 잘 되어 있어야 하며, 각 함수에는 주석을 달아줘.

