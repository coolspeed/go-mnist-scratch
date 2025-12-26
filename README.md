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
├── bin/                # 컴파일된 바이너리 (생성됨)
├── cmd/                # 애플리케이션 진입점
│   ├── benchmark/      # 성능 벤치마킹 도구
│   ├── server/         # 웹 서버 및 추론 API
│   ├── train/          # 모델 학습 실행
│   └── validate/       # 모델 정확도 검증 도구
├── data/               # MNIST 데이터셋 파일 (다운로드됨)
├── docs/               # 문서 (성능 분석 등)
├── matrix/             # 행렬 연산 라이브러리 (직접 구현)
├── neural/             # 신경망 모델 정의 및 학습/추론 로직
├── static/             # 웹 프론트엔드 파일 (HTML/JS)
├── utils/              # 유틸리티 (데이터 로더 등)
├── Makefile            # 빌드 및 실행 자동화
└── setup.sh            # 데이터 다운로드 스크립트
```

## 신경망 아키텍처

- **Input Layer**: 784 노드 (28x28 픽셀)
- **Hidden Layer**: 200 노드 (Sigmoid 활성화 함수)
- **Output Layer**: 10 노드 (Softmax 활성화 함수)

## 시작하기

MNIST 데이터셋은 `data/` 디렉토리에 이미 포함되어 있어 별도의 다운로드 없이 바로 시작할 수 있습니다.

### 1. 학습 (Training)

```bash
make train
# 또는
go run cmd/train/main.go
```

학습된 모델은 `mnist_model.gob` 파일로 저장됩니다.

### 2. 검증 (Validation)

학습된 모델의 정확도를 검증합니다.

```bash
make validate
# 또는
go run cmd/validate/main.go
```

### 4. 추론 서버 실행 (Inference Server)

웹 인터페이스를 통해 손글씨를 직접 그려서 테스트할 수 있습니다.

```bash
make server
# 또는
go run cmd/server/main.go
```

브라우저에서 `http://localhost:8080` 접속

## 학습 파라미터

- **Learning Rate**: 0.3
- **Epochs**: 20
- **Batch Size**: 64
- **Loss Function**: Cross-Entropy Loss
- **Model File**: mnist_model.gob

## 개발 명령어 (Makefile)

- `make build`: 서버 바이너리 빌드
- `make train`: 모델 학습
- `make server`: 서버 실행
- `make validate`: 모델 검증
- `make test`: 유닛 테스트 실행
- `make clean`: 빌드된 파일 및 로그 정리

## 참고 자료

- [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)
