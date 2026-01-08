# DeepGastro: Federated Learning for Gastrointestinal Endoscopy
**(내시경 진단을 위한 프라이버시 보존형 연합학습 시스템)**

> **Target Organs:** Stomach (위), Colon (대장)  
> **Diagnosis Classes:** Ulcer(궤양), Cancer(암), Polyp(용종)

---

## 프로젝트 개요 (Overview)

**DeepGastro**는 다기관(Hospital)이 보유한 내시경 이미지 데이터를 **외부로 반출하지 않고(Privacy-Preserving)**, 각 병원 내부에서 학습한 **가중치(Weights)만을 공유**하여 고성능 AI 모델을 만드는 **Cross-Silo Federated Learning(연합학습)** 시스템입니다.

본 프로젝트는 의료 데이터의 민감성(Non-IID, 보안 규제)을 해결하기 위해 **FedAvg (Federated Averaging)** 알고리즘을 기반으로 구축되었으며, 실제 임상 환경(Backend)에 즉시 투입 가능한 **추론(Inference) 모듈**을 포함하고 있습니다.

---

## 시스템 아키텍처 (System Architecture)

본 시스템은 **Central Server(중앙 서버)**와 **Client Hospitals(참여 병원)** 간의 **Hub-and-Spoke** 구조로 동작합니다.

### 1. 학습 프로세스 (Federated Training Loop)
1.  **Global Model Distribution:** 서버가 최신 글로벌 모델(`global_model_broadcast.pth`)을 각 병원에 배포합니다.
2.  **Local Training:** 각 병원은 자신의 **로컬 데이터(Private Data)**를 사용하여 모델을 학습시킵니다. (데이터는 절대 병원 밖으로 나가지 않음)
3.  **Weight Upload:** 학습된 모델의 가중치(Weights) 파일만 서버로 전송합니다.
4.  **Aggregation (FedAvg):** 서버는 수집된 가중치들을 **평균(Averaging)** 내어 새로운 글로벌 모델을 생성합니다.
5.  **Iteration:** 위 과정을 목표 성능에 도달할 때까지 반복합니다.

### 2. 배포 프로세스 (Inference & Deployment)
- 학습이 완료된 가중치는 `src/inference.py` 모듈을 통해 **백엔드 서버**에 즉시 통합됩니다.
- **Auto-Device Detection:** 실행 환경(NVIDIA GPU, Apple Silicon MPS, CPU)을 자동으로 감지하여 최적의 성능을 냅니다.

---

## 기술 스택 (Tech Stack)

- **Language:** Python 3.9+
- **Framework:** PyTorch, Torchvision
- **Algorithm:** FedAvg (Federated Averaging)
- **Model Architecture:** EfficientNet-B0 (Modified Classifier)
- **Deployment:** Custom Inference Module (Integratable with FastAPI/Django/Spring)

---

## 📂 디렉토리 구조 (Directory Structure)

```bash
deepgastro/
├── data/                      # (보안) 학습 데이터 (Git 제외됨)
├── saved_models/              # 학습된 모델 가중치 저장소
│   ├── colon/
│   │   ├── main_weights/      # 서버가 배포하는 Global Model
│   │   └── client_weights/    # 병원이 제출하는 Local Update
│   └── stomach/               # (위장 모델도 동일 구조)
├── src/                       # 소스 코드
│   ├── client.py              # 병원 측 학습 로직 (GastroClient)
│   ├── server.py              # 서버 측 통합 로직 (GastroServer)
│   ├── inference.py           # [핵심] 백엔드 연동용 진단 모듈
│   ├── run_client.py          # [실행] 병원 학습 스크립트
│   ├── run_server_aggregation.py # [실행] 서버 통합 스크립트
│   ├── models/                # 모델 아키텍처 정의
│   └── data/                  # 데이터셋 로더 (Dataset Class)
├── requirements.txt           # 의존성 패키지 목록
└── README.md                  # 프로젝트 설명서
```

---

## 🚀 설치 및 실행 방법 (How to Run)

### 1. 환경 설정 (Installation)

```bash
# 저장소 클론
git clone [https://github.com/DeepGastro/b-fl-ai-research.git](https://github.com/DeepGastro/b-fl-ai-research.git)
cd b-fl-ai-research
```

### 2. 패키지 설치
```
pip install -r requirements.txt
```

### 3. 연합학습 실행 (Federated Learning Simulation)

**Step A. 병원(Client) 학습 시작**
병원 A가 **대장(Colon)** 혹은 **위장(Stomach)** 데이터를 이용해 학습을 수행합니다.
```bash
python -m src.run_client --id hospital_a --organ colon
```

**Step B. 중앙 서버(Server) 통합**
모든 병원의 학습이 끝나면 서버 관리자가 통합을 수행합니다.
```bash
python -m src.run_server_aggregation
```

---

## 🔌 백엔드 연동 가이드 (For Backend Developers)

이 시스템은 웹/앱 서버(Backend)에서 AI 진단 기능을 쉽게 사용할 수 있도록 **독립적인 추론 모듈 (`src/inference.py`)**을 제공합니다.

### 특징
- **의존성 최소화:** `src` 폴더만 있으면 어디서든 동작.
- **하드웨어 가속:** `CUDA` (Linux Server), `MPS` (MacBook), `CPU` 자동 전환.
- **입력 유연성:** 파일 경로(`str`) 또는 이미지 객체(`PIL.Image`) 모두 지원.

### 사용 예시 코드
백엔드 서버 코드에서 다음과 같이 호출하여 사용할 수 있습니다.

```python
from src.inference import GastroPredictor

# 1. 진단기 초기화 (서버 시작 시 1회 실행 권장)
# organ 옵션: 'colon' 또는 'stomach'
ai_doctor = GastroPredictor(target_organ="colon")

# 2. 이미지 진단 (요청 들어올 때마다 실행)
# 이미지 파일의 경로만 넘겨주면 됩니다.
image_path = "uploads/patient_1024.jpg"
result_text = ai_doctor.predict_image(image_path)

# 3. 결과 출력
print(result_text)
# 출력 예시: "암 (확률: 98.52%)"
```

---

## FedAvg 알고리즘 상세 (Algorithm Details)

본 프로젝트에 적용된 알고리즘의 수식적 배경은 다음과 같습니다.

1. **Local Update:** 각 병원 $k$는 자신의 데이터셋 $D_k$를 사용하여 손실 함수 $L$을 최소화하는 방향으로 로컬 가중치 $w_k$를 업데이트합니다.
   $$w_{k}^{t+1} \leftarrow w_k^t - \eta \nabla L(w_k^t; D_k)$$

2. **Global Aggregation:** 서버는 $K$개의 병원으로부터 가중치를 수집하여 산술 평균을 구합니다.
   $$w_{global}^{t+1} = \frac{1}{K} \sum_{k=1}^{K} w_k^{t+1}$$

이 방식을 통해 중앙 서버는 **데이터를 직접 보지 않고도(Privacy)** 전체 데이터 분포를 학습한 것과 유사한 성능을 얻을 수 있습니다.

---

## 보안 및 주의사항 (Security Note)

- **데이터 보안:** 본 리포지토리에는 환자의 개인정보가 포함된 **이미지 데이터(Raw Data)는 포함되어 있지 않습니다.** (`.gitignore` 적용됨)
- **모델 가중치:** 학습된 `.pth` 파일은 보안 및 용량 문제로 업로드되지 않으며, 재현성을 위한 **Pretrained 초기 모델**만 제공됩니다.

---

## Maintainers

- **AI Part Lead:** [Jadest03] (Lead Researcher & Architect)
- **Organization:** DeepGastro Research Team