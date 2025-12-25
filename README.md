# Go-MNIST-Scratch

순수 Go만 사용하여 MNIST 손글씨 숫자 분류 신경망을 밑바닥부터 구현하는 프로젝트입니다. 외부 ML 라이브러리 없이 3층 신경망(Input-Hidden-Output)을 직접 구현하여 학습 및 추론을 수행합니다.

## 목표

- **Pure Go**: 행렬 연산부터 신경망까지 모두 Go 표준 라이브러리와 `[][]float64` 슬라이스로 직접 구현
- **경량화**: CPU에서 빠르게 동작하는 초경량 신경망 모델
- **학습**: MNIST 데이터셋으로 90% 이상 정확도 달성
- **추론**: 단일 예측 < 10ms 속도 목표

## 프로젝트 구조

```plaintext
go-mnist-scratch/
├── data/               # MNIST 데이터셋 파일 (idx1-ubyte, idx3-ubyte)
├── matrix/             # 행렬 연산 라이브러리 (직접 구현)
│   ├── matrix.go
│   └── matrix_test.go
├── neural/             # 신경망 모델 정의 및 학습/추론 로직
│   ├── network.go
│   └── activations.go
├── utils/              # MNIST 파일 파싱 및 이미지 처리 유틸리티
│   └── loader.go
├── main.go             # 학습 실행 및 모델 저장
├── server.go           # 웹 서버 및 추론 API
└── setup.sh            # 데이터 다운로드 및 설정 스크립트
```

## 신경망 아키텍처

- **Input Layer**: 784 노드 (28x28 픽셀)
- **Hidden Layer**: 200 노드 (Sigmoid 활성화 함수)
- **Output Layer**: 10 노드 (Softmax 활성화 함수)

## 시작하기

### 1. 데이터 다운로드

```bash
./setup.sh
```

### 2. 학습

```bash
go run main.go
```

학습된 모델이 `model.json`으로 저장됩니다.

### 3. 추론 서버 실행

```bash
go run server.go
```

## 학습 파라미터

- **Learning Rate**: 0.3
- **Epochs**: 20
- **Batch Size**: 64
- **Loss Function**: Cross-Entropy Loss

## 참고 자료

- [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)
