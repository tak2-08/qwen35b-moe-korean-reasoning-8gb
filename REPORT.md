# Qwen3.6-35B-A3B 한국어 단계별 추론 특화: 8GB GPU 한 장으로 35B MoE 모델 학습하기

기술 보고서 (2026-11) · 라이선스: CC BY-NC 4.0 · 이 저장소에는 모델 가중치와 학습 데이터가 들어 있지 않습니다.

---

## English summary

We adapt Qwen3.6-35B-A3B (35B-parameter Mixture-of-Experts, 3B active, 256 routed experts per layer), quantized to
IQ3_S (12.7 GB), to Korean multiple-choice exam reasoning on a single consumer PC: **RTX 3070 8 GB, 16 GB system
RAM, Ryzen 5 5600X, NVMe SSD, Windows**. The model does not fit in GPU memory, and a conventional fine-tuning
stack cannot even load it.

Method:
1. **Frozen-routed-expert surrogate.** The real inference engine (llama.cpp) runs the model as usual and records
   every routed-expert output. A PyTorch re-implementation of the attention, Gated DeltaNet, norm and
   shared-expert parts (no routed experts) then treats those recorded outputs as constants. This gives exact
   gradients through every layer above the edited one, while the 12.7 GB of expert weights never enters GPU memory.
2. **Three memory tiers.** The GPU holds the non-expert layers and the LoRA. System RAM holds the expert weights
   for the engine. The SSD holds the activation capture: about 125 KB per token, about 70 GB for one 460k-token
   chunk. Training runs chunk by chunk: capture, one epoch, delete the capture.
3. **Chunked Gated DeltaNet.** A 64-token chunked (WY/UT-transform) form of the gated delta rule matches the
   token-by-token reference within 1e-6, gradients included. It made training about **40× faster**: one epoch
   went from 3.6 h to 7 min.
4. **Forgetting control.** KL replay to the unedited model on background text: weight 12, 128-token windows.
   Every checkpoint must pass a real-engine gate: the edited model must give the same top-1 next token as the
   original model on at least 92% of held-out Wikipedia tokens.

Data: about 2,900 private Korean step-by-step (CoT) solutions, synthesized with multiple teacher LLMs and
filtered by automatic answer matching and repetition-loop filtering. Each solution's final answer had to equal
the official answer key, and looping solutions were dropped. The newest exam year was never used for training and
serves only as the test set.

The adapter is a LoRA on the shared-expert down projection, layers 12–39, rank 16 (4.6 MB GGUF).

Results on the held-out exam year:

| | Base | Stage-1 adapter (c1a) | **Adopted c2** |
|---|---|---|---|
| Step-by-step, 220 items, ≤2000 tokens (accuracy) | 53.2% | — | **63.6%** (p = 0.006 vs base, paired) |
| Step-by-step, first 120 items (development set) | 52.5% | 52.5% | **63.3%** (p = 0.041) |
| Average answer length (tokens; 220 / 120 / 300 items) | ~1,330 | 371 | ~540 |
| Answers cut off at 2000 tokens (220 items; c1a: 120) | 30.5% | 5.8% | 5.0% |
| Direct answer (no reasoning), 1,416 items | **62.8%** | 57.4% | 60.5% |
| Korean benchmark mix, 925 items | **62.7%** | 61.3% | 61.6% |
| Background next-token agreement (gate ≥ 92%) | 100% | 93.2% | 92.4% |

Adopted c2 = the checkpoint after 3 of 5 chunks, used with frequency/presence penalty 0.3. The final checkpoint
fell to 91.3% agreement and was rejected by the gate.

Honest reading:
- The adapter makes step-by-step answers more accurate and much shorter: about 2.4× fewer tokens than the base model
  under the same 2000-token limit.
- It slightly lowers accuracy when the model must answer immediately without reasoning. The app therefore
  enables it only for multiple-choice exam questions.
- The step-by-step gain was first found on 120 items (p = 0.041), then confirmed on 100 more items:
  220 items in total, p = 0.006. On all 300 held-out items the adopted model scores 63.7%.

---

## 0. 한눈에 보기

- **무엇을**: 350억 개 파라미터 MoE 모델(Qwen3.6-35B-A3B, IQ3_S 12.7GB)을 한국어 객관식 시험 문제의 단계별 풀이에 맞게 추가 학습했습니다.
- **어디서**: 일반 가정용 PC 한 대입니다.
  - GPU: RTX 3070 8GB
  - RAM: 16GB
  - CPU: Ryzen 5 5600X
  - 저장장치: NVMe SSD
  - OS: Windows
- **어떻게**:
  - 모델이 GPU에 다 들어가지 않습니다.
  - 그래서 추론 엔진(llama.cpp)으로 전문가(expert) 출력을 기록해 두고, 나머지 부분만 PyTorch로 다시 구현했습니다.
  - 이렇게 해서 정확한 기울기로 학습했습니다.
- **결과**: 학습에 한 번도 쓰지 않은 최신 연도 시험 문항에서 측정했습니다.
  - 단계별 풀이 정답률(같은 220문항): 53.2% → **63.6%** (p = 0.006)
  - 풀이 길이: 평균 약 1,330토큰 → 약 540토큰 (약 2.4배 짧아짐)
  - 일반 문장에 대한 원래 모델과의 일치도: 92.4% (기준 92% 통과)
- **한계**: 풀이 없이 바로 답을 고르는 정답률은 기본 모델보다 2.3%p 낮습니다. 그래서 실제 앱에서는 객관식 시험 문제에만 이 어댑터를 켭니다.

## 1. 문제: 8GB GPU에서 35B MoE를 학습할 수 없다

| 구성 요소 | 크기 | 8GB GPU에 들어가는가 |
|---|---|---|
| 모델 전체 (IQ3_S) | 12.7GB | 아니오 |
| 라우팅 전문가 (40층 × 256개) | 약 12GB | 아니오 |
| 나머지 (어텐션, DeltaNet, 공용 전문가, 임베딩) | 약 1GB | 예 |

- 일반적인 LoRA 학습 도구(PEFT 등)는 모델 전체를 GPU 또는 RAM에 펼쳐 놓아야 합니다.
  - 16비트로 풀면 약 70GB입니다.
  - 이 PC의 RAM 16GB로는 불러오는 것조차 불가능합니다.
- 추론은 llama.cpp가 전문가 일부를 CPU에서 계산하는 방식으로 가능합니다. 하지만 llama.cpp에는 학습 기능이 없습니다.

## 2. 방법

### 2-1. 고정 전문가 대리 모델 (Frozen-routed-expert surrogate)

핵심 관찰:
- 학습할 부분을 공용 전문가(shared expert)로 한정하면, 라우팅 전문가는 학습하지 않습니다.
- 그렇다면 라우팅 전문가의 **출력**만 알면 됩니다. 가중치는 필요 없습니다.

절차는 다음과 같습니다.
1. **기록 (`extract_surr.py`, `hm_host.cpp`)**
   - 실제 엔진이 학습 문장을 평소처럼 처리합니다.
   - 이때 12층으로 들어가는 잔차(residual)와 12 - 39층 각각의 라우팅 전문가 출력(`ffn_moe_out`)을 SSD에 기록합니다.
   - `hm_host.cpp`는 llama.cpp의 공개 API(콜백)만 사용합니다. 엔진 자체는 고치지 않았습니다.
2. **대리 모델 (`qwen_torch.py`)**
   - 12 - 39층의 어텐션, Gated DeltaNet(선형 어텐션), 정규화, 공용 전문가를 PyTorch로 다시 구현했습니다.
   - 수치는 llama.cpp와 맞췄습니다. 검증은 `valid_torch.py`로 합니다.
   - 라우팅 전문가 자리에는 1단계에서 기록한 값을 상수로 넣습니다.
3. **학습 (`train_surr.py`)**
   - 12 - 39층 공용 전문가의 down 행렬에 LoRA(rank 16)를 붙여 학습합니다.
   - 어텐션, DeltaNet, 정규화를 거치는 기울기는 근사 없이 정확합니다.
   - 근사는 하나뿐입니다. 위쪽 층의 라우팅 전문가 출력이 입력 변화에 따라 바뀌는 효과를 무시합니다.
4. **보정 (refresh)**
   - 위 근사의 오차가 쌓이지 않도록, 다음 묶음은 **직전 묶음의 어댑터를 적용한 상태로 다시 기록**한 뒤 학습합니다.
5. **실제 엔진 검증 (`eval_mid.py`, `eval_exam.py`)**
   - 학습이 끝난 어댑터를 llama.cpp LoRA 형식(GGUF)으로 내보냅니다.
   - 그다음 실제 엔진에 올려서 모든 지표를 다시 잽니다.
   - 대리 모델의 숫자는 참고용이고, 채택 여부는 실제 엔진 숫자로만 정합니다.

### 2-2. 3단 메모리 계층

| 계층 | 담는 것 | 크기 |
|---|---|---|
| GPU (8GB) | 대리 모델 가중치(12 - 39층, 전문가 제외), LoRA, 활성값 | 약 6 - 7GB |
| RAM (16GB) | 기록 단계에서 엔진이 쓰는 전문가 가중치 | 약 12GB |
| SSD | 기록된 활성값: 토큰당 약 125KB (2048차원 × 반정밀도 × 약 31개 텐서) | 묶음당 약 70GB |

- 학습 데이터 200만 토큰을 한 번에 기록하면 약 250GB가 필요합니다.
- 그래서 약 46만 토큰씩 5개 묶음으로 나누고, 묶음마다 "기록 → 1회 학습 → 기록 삭제"를 반복했습니다.
- 묶음 하나에 기록 약 23분, 학습 약 14분이 걸립니다.

### 2-3. Gated DeltaNet 묶음 계산: 학습 40배 가속

- Qwen3.6의 층 대부분은 선형 어텐션(Gated DeltaNet)입니다.
- 이 점화식을 토큰 하나씩 계산하는 기준 구현은 GPU에서 매우 느렸습니다.
- 64토큰 묶음 단위의 WY/UT 변환 형태로 바꿨습니다.
  - 묶음 안의 계산은 모두 행렬곱입니다.
  - 순서대로 계산해야 하는 것은 묶음 사이의 상태 전달뿐입니다.
- 기준 구현과 출력 및 기울기 오차는 1e-6 이내입니다.
- 학습 1회차 시간이 **3.6시간 → 7분(약 40배)**으로 줄었습니다.

### 2-4. 학습 데이터

- **구성**:
  - 다중 교사 모델(Multi-Teacher LLM)을 활용해 구축했습니다.
  - 자동 정답 대조와 반복 루프 필터링을 거쳤습니다.
  - 비공개 한국어 단계별 추론(CoT) 합성 데이터 약 2,900건입니다.
- **정답 대조**: 풀이의 마지막 "정답"이 공식 정답과 같은 것만 남겼습니다.
- **반복 필터 (`cot_filter.py`)**: 다음 중 하나라도 해당하면 버렸습니다.
  - 40자 이상 문장이 그대로 2번 이상 나오거나, 어떤 문장이든 6번 이상 나옴
  - 30자 구간이 3번 이상 반복됨
  - "다시 확인", "검산" 같은 자기 점검 표현이 3번 이상 나옴
  - 정답이 중간에 바뀜
- **규모**: 문항당 풀이 1개, 2,903개 시퀀스, 약 200만 토큰이며 최대 2,000토큰입니다.
- **섞어 넣은 데이터**: 묶음마다 약 12%는 1단계(c1) 학습 데이터입니다. 기출 공식 해설과 뉴스 문장으로, 앞서 배운 내용을 유지하기 위해 넣었습니다.
- **평가 분리**: 가장 최근 시험 연도의 문항은 학습과 검색 자료 어디에도 넣지 않았습니다. 평가에만 썼습니다.
- **데이터 공개**: 학습 데이터와 수집 과정은 공개하지 않습니다.

### 2-5. 망각 제어와 체크포인트 선택

- **KL 재생**: 위키백과 배경 문장에서 수정된 모델의 출력 분포가 원래 모델과 같아지도록 KL 손실을 더했습니다.
  - 가중치: 12
  - 문장 길이: 128토큰
  - 1단계에서 가중치를 4에서 12로 올리자, 일치도가 87.8%에서 93.2%로 회복됐습니다.
- **사전에 정한 채택 조건 4가지**: 모두 실제 엔진에서 잽니다.
  1. 배경 문장 일치도 ≥ 92%. 원래 모델과 같은 다음 토큰을 고르는 비율이며, 학습에 쓰지 않은 위키백과 토큰으로 잽니다.
  2. 단계별 풀이 정확도가 이전 어댑터(c1a)보다 높음
  3. 바로 답하기 정확도가 c1a보다 2%p 넘게 낮지 않음
  4. 한국어 벤치마크가 c1a보다 1.5%p 넘게 낮지 않음
- **실제로 일어난 일**:
  - 묶음을 이어 학습할수록 일치도가 조금씩 내려갔습니다. 대리 모델 기준, 각 묶음 학습 직후 값입니다.
    - 묶음 0: 97.9%
    - 묶음 2: 93.5%
    - 묶음 4: 92.4%
  - 실제 엔진 기준으로는 최종(5묶음) 체크포인트가 91.3%로 조건 1에 미달했습니다.
  - 3묶음 체크포인트는 92.4%로 4가지 조건을 모두 통과해 채택했습니다.
  - 정확도도 3묶음 쪽이 더 높았습니다(단계별 61.7% 대 55.0%).
  - 즉, 오래 학습할수록 좋아지지 않았습니다. 중간 지점이 정확도와 보존의 균형이 가장 좋았습니다.

## 3. 정량 결과

모든 수치는 실제 llama.cpp 엔진, 같은 모델 파일, 생각(thinking) 모드를 끈 상태에서 잰 것입니다. 평가 문항은 학습에 쓰지 않은 가장 최근 연도의 시험입니다. 그림이 필요한 문항은 제외했습니다.

### 3-1. 단계별 풀이 (같은 120문항, 답 최대 2,000토큰, greedy)

| 모델 | 정확도 | 정답 형식으로 끝맺음 | 반복 발생 (같은 줄 3회 이상) | 길이 초과로 잘림 | 평균 길이 |
|---|---|---|---|---|---|
| 기본 모델 | 52.5% | 74.2% | 9.2% | 32.5% | 1,388토큰 |
| c1a (1단계, 공식 해설 학습) | 52.5% | 89.2% | 2.5% | 5.8% | 371토큰 |
| c2 최종 (5묶음) | 55.0% | 92.5% | 5.8% | 5.8% | 537토큰 |
| c2 최종 + 빈도 페널티 0.3 | 56.7% | 95.8% | 2.5% | 3.3% | 506토큰 |
| c2 3묶음 | 61.7% | 94.2% | 6.7% | 5.8% | 562토큰 |
| **c2 3묶음 + 빈도 페널티 0.3 (채택)** | **63.3%** | 95.0% | 5.0% | 4.2% | **524토큰** |

**통계 검정**: 같은 문항 쌍에 대한 McNemar 정확 검정입니다.
- 채택 모델 대 기본 모델: 76/120 대 63/120. 채택 모델만 맞힌 문항 24개, 기본 모델만 맞힌 문항 11개, p = 0.041
- 채택 모델 대 c1a: p = 0.035
- 유의하지만 표본이 작아서, 오차 범위는 대략 ±8%p입니다.

**추가 검증 (220문항)**: 위 비교로 설정을 고른 뒤, 뒤쪽 100문항(121~220번)을 기본 모델과 채택 모델로 새로 풀려 합쳤습니다.

| 모델 | 정확도 (220문항) | 2,000토큰에서 잘림 | 평균 길이 |
|---|---|---|---|
| 기본 모델 | 53.2% (117/220) | 30.5% | 약 1,330토큰 |
| **채택 c2** | **63.6% (140/220)** | **5.0%** | 약 540토큰 (300문항 평균 544) |

- McNemar 정확 검정: 채택 모델만 맞힌 문항 44개, 기본 모델만 맞힌 문항 21개, **p = 0.006**
- 새로 푼 100문항만 보면 기본 54.0%, 채택 64.0%로, 앞 120문항과 같은 방향·비슷한 크기의 차이가 다시 나왔습니다.
- 채택 모델은 보류 문항 300개 전체에서 63.7%(평균 544토큰)입니다.

**참고: 기본 모델에 생각 모드를 켠 경우**
- 앞쪽 60문항, 생각 예산 3,000토큰입니다.
- 정확도 48.3%, 평균 2,970토큰입니다.
- 채택 모델은 생각 모드 없이 이보다 높은 정확도를 평균 524토큰으로 냅니다.
- 다만 문항 수(60 대 120)와 모드가 달라서, 이 비교는 참고용입니다.

### 3-2. 바로 답하기 (풀이 없이 정답 번호 확률 비교, 전체 1,416문항)

| 모델 | 정확도 |
|---|---|
| **기본 모델** | **62.8%** |
| c1a | 57.4% |
| c2 최종 | 58.4% |
| c2 3묶음 (채택) | 60.5% |

해석:
- 풀이 없이 즉답할 때는 어댑터가 오히려 불리합니다.
- 어댑터는 "풀이를 쓰고 결론을 내는" 습관을 학습했기 때문으로 보입니다.
- 그래서 실제 사용에서는 단계별 풀이 모드에서만 켭니다.

과목별(채택 모델, 정답/문항):

| 국어 | 영어 | 한국사 | 사회탐구 | 과학탐구 | 수학 |
|---|---|---|---|---|---|
| 304/383 (79%) | 51/63 (81%) | 64/102 (63%) | 353/623 (57%) | 59/157 (38%) | 26/88 (30%) |

### 3-3. 한국어 공개 벤치마크 (925문항, 바로 답하기)

구성:
- KMMLU 12과목 × 30문항
- HAE-RAE Bench 8영역 × 40문항
- 2025학년도 수능 국어 45문항
- 공무원 시험 기출

| 모델 | 전체 | HAE-RAE | KMMLU | 수능 국어 2025 | 공무원 |
|---|---|---|---|---|---|
| 기본 모델 | **62.7%** | **70.6%** | **58.6%** | 75.6% | 54.5% |
| c1a | 61.3% | 66.2% | 57.2% | 82.2% | **56.0%** |
| c2 최종 | 62.1% | 70.0% | 57.2% | 82.2% | 53.5% |
| c2 3묶음 (채택) | 61.6% | 69.4% | 58.1% | **82.2%** | 51.0% |

- 수능 국어는 +6.6%p(45문항 중 3문항 차이) 올랐습니다.
- 나머지는 기본 모델과 비슷하거나 약간 낮습니다. 일반 지식이 크게 망가지지 않았다는 근거로 봅니다.

### 3-4. 일반 문장 보존 (실제 엔진, 위키백과 문장의 다음 토큰이 원래 모델과 같은 비율)

| 모델 | 일치도 |
|---|---|
| c1a | 93.2% |
| **c2 3묶음 (채택)** | **92.4%** |
| c2 최종 | 91.3% (기준 미달) |

### 3-5. 속도와 메모리 (채팅 앱 실측, 어댑터 적용)

| 항목 | 값 |
|---|---|
| 생성 속도 | 초당 약 52 - 55토큰 (그 전에 대화가 없던 첫 요청은 약 35) |
| VRAM 사용 | 약 7.7GB / 8GB |
| 엔진 프로세스 RAM | 약 8.5 - 9.0GB |

서빙 설정:
- `--n-cpu-moe 32`: 32개 층의 전문가를 CPU에 둡니다.
- llama.cpp 내장 GPU 전문가 캐시: 2,000MiB이며, 캐시에 없는 전문가의 80%는 CPU가 계산합니다.
- 이 캐시 기능은 llama.cpp에 원래 들어 있는 기능이고, 이 저장소에서는 값만 조정했습니다. 조정으로 초당 38 → 59.5토큰(256토큰 생성 기준, 어댑터 없음)이 됐습니다.

## 4. 사례: 최신 연도 수학 공통 15번 (4점)

1단계 어댑터(c1a)가 실사용 중 같은 문장을 끝없이 반복했던 문항입니다. 저작권 때문에 문제 원문은 싣지 않고 문항 번호만 적습니다. 정답은 ③입니다.

| 모델과 설정 | 실행 횟수 | 무한 반복 | ③이라고 답함 | 그중 풀이 과정이 실제로 정답에 이른 경우 |
|---|---|---|---|---|
| c1a: greedy와 온도 0.8 × 반복 억제 설정 4가지 | 8 | 앱에서 발생 (4,096토큰 시험에서는 반복 없이 끝남) | 0 | — |
| 채택 c2: 채팅 앱 경유, greedy + 시드 3개 × 설정 3가지 | 12 | **0** (모두 정상 종료, 최대 2,643토큰) | 5 | **1** |

해석:
- **반복은 해결됐습니다.** 12번 모두 답을 끝까지 쓰고 "정답:" 줄로 마쳤습니다.
- **하지만 제대로 푼 것은 아닙니다.** ③이라고 답한 5번 가운데 자기 풀이로 8을 실제로 구한 것은 1번뿐입니다.
- 나머지 4번은 11, 5, −1 같은 다른 값을 계산하고도 ③을 적었습니다. "주어진 정답은 ③" 같은 문장을 쓴 경우도 있었습니다.
- 지시문의 형식 예시("정답: ③")가 이 문항의 정답과 우연히 같아서 생긴 일로 보입니다.
- greedy(결정적) 디코딩에서는 ①을 답합니다.
- 결론: 이 문항에서 반복 없이 끝맺게 됐지만, 문제 자체는 아직 풀지 못합니다.

**전체 평가의 편향 점검**: 모든 단계별 풀이 평가가 같은 형식 예시를 쓰기 때문에, ③을 지나치게 고르는지 확인했습니다. (220문항 결과, 괄호 안은 처음 120문항)

| 모델 | ③을 고른 횟수 (실제 정답 ③: 51 (30)) | 정답이 ③인 문항 정확도 | 정답이 ③이 아닌 문항 정확도 |
|---|---|---|---|
| 기본 모델 | 78 (41) | 74.5% (70.0%) | 46.7% (46.7%) |
| c1a | — (30) | — (56.7%) | — (51.1%) |
| 채택 c2 | 57 (31) | 74.5% (66.7%) | **60.4% (62.2%)** |

- 기본 모델은 ③ 쪽으로 쏠려 있어서 점수가 다소 부풀려져 있습니다.
- 채택 모델은 이런 쏠림이 없습니다. 정답이 ③이 아닌 문항만 보면 기본 모델과의 차이가 10.5%p에서 13.7%p로 더 커집니다(220문항 기준).
- "공식 정답"이나 "주어진 정답"을 언급한 문장은 채택 모델의 120개 답 중 1개에만 나왔고, 그 답은 틀렸습니다.


## 5. 권장 추론 설정

| 항목 | 값 | 이유 |
|---|---|---|
| 생각 모드 | 끔 (`enable_thinking: false`) | 어댑터는 생각 모드 없는 단계별 풀이 형식으로 학습됐습니다. 생각 모드를 켜면 오히려 나빠졌습니다. |
| frequency_penalty / presence_penalty | 0.3 / 0.3 | 120문항에서 정확도 +1.6%p, 반복 6.7% → 5.0%, 잘림 5.8% → 4.2% |
| max_tokens | 2,000 이상 | 2,000에서 잘리는 비율 4.2% |
| 프롬프트 끝 | "단계별로 풀고 마지막 줄에 '정답: ③' 형식으로 답하세요." | 학습 형식 그대로 |
| repetition_penalty | 쓰지 않음 | 1.1에서 "정답:" 줄 같은 필수 형식까지 억제되는 사례가 있었습니다. |

실행 예시 (llama.cpp):

```
llama-server -m Qwen3.6-35B-A3B-UD-IQ3_S.gguf --lora exam_adapter.gguf \
  -ngl 99 --n-cpu-moe 32 -c 8192 -fa on --jinja -t 6 --host 127.0.0.1 --port 8080
```

## 6. 한계

- **표본 크기**: 단계별 풀이 비교는 220문항(p = 0.006)입니다. 설정은 앞 120문항으로 골랐고, 뒤 100문항은 확인용으로 새로 풀었습니다. 정확도 차이의 오차 범위는 대략 ±6%p입니다.
- **즉답 정확도 하락**: 바로 답하기에서는 기본 모델보다 낮습니다(62.8% 대 60.5%).
- **보존 지표의 범위**: "일반 문장 보존"은 위키백과 문장의 다음 토큰 일치도입니다. 대화 품질 전체를 직접 잰 것은 아닙니다.
- **과목 편차**: 수학(30%)과 과학탐구(38%)는 여전히 낮습니다. 그림이 필요한 문항은 평가에서 뺐습니다.
- **반복 실험 없음**: 학습은 한 번만 돌렸습니다. 시드를 바꿔 반복하지 않았습니다.
- **공개 범위**: 학습 데이터와 어댑터 가중치는 공개하지 않습니다. 이 저장소는 방법과 코드, 측정 결과를 공개합니다.

## 7. 코드 구성 (`src/`)

| 파일 | 역할 |
|---|---|
| `hm_host.cpp` | llama.cpp 공개 API로 모델을 띄워 두고, 요청마다 토큰별 은닉 상태와 지정한 내부 텐서(전문가 출력 등)를 돌려주는 도구 |
| `host.py` | 위 도구의 Python 클라이언트 |
| `qwen_torch.py` | Qwen3.6 MoE 디코더 층의 PyTorch 재구현 (라우팅 전문가 제외, 64토큰 묶음 DeltaNet) |
| `valid_torch.py` | 재구현이 실제 엔진과 같은 값을 내는지 검증 |
| `cot_filter.py` | 반복, 순환 풀이 필터 |
| `build_c2_cache.py` | 학습 묶음 토큰 캐시 생성 (데이터 파일 형식은 파일 머리말 참고) |
| `extract_surr.py` | 묶음별 활성값 기록 (직전 어댑터 적용 가능) |
| `make_hbase.py` | 배경 문장에 대한 원래 모델의 출력 (KL 기준값) |
| `train_surr.py` | 대리 모델 학습, LoRA를 llama.cpp GGUF로 내보내기 |
| `eval_mid.py` | 실제 엔진에서 배경 문장 일치도 측정 |
| `eval_exam.py` | 실제 엔진에서 시험 정확도 측정 (바로 답하기, 단계별 풀이, 반복률, 샘플링 설정) |
| `run_c2.sh` | 전체 학습 파이프라인 |

경로는 환경변수로 지정합니다.

| 환경변수 | 가리키는 것 |
|---|---|
| `MODEL` | GGUF 모델 파일 |
| `ENGINE` | llama-server 실행 파일 |
| `HM_DATA` | 활성값 기록 폴더 |
| `EXAM_DATA` | 시험 데이터 폴더 |
| `RESULTS` | 결과 폴더 |
| `LAB` | `bin/hm_host.exe`가 있는 폴더 |

## 8. 라이선스

- 이 저장소의 코드, 문서, 보고서: **CC BY-NC 4.0** (비상업적 연구와 학술 목적 전용)
- 기본 모델(Qwen3.6-35B-A3B): 원 저작자의 라이선스를 따릅니다.
- llama.cpp: MIT 라이선스
