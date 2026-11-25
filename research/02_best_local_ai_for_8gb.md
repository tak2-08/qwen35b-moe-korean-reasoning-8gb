# 2차 연구: 이 PC에 가장 알맞은 로컬 AI 찾기 + 설치

작성: 2026-11-22 · 대상 PC: RTX 3070 8GB / RAM 16GB / Ryzen 5 5600X

## 결론

**Qwen3.6-35B-A3B (UD-IQ3_S, 12.7GB) + llama.cpp 내장 채팅 화면**을 골랐고, 설치까지 마쳤어요.

| 항목 | 최종 측정값 (실제 UI 기준) |
|---|---|
| 답변 생성 속도 | **초당 35~37토큰** |
| 질문을 보내고 답이 시작될 때까지 | **약 1.3초** |
| 켜는 데 걸리는 시간 | 약 35초 (모델 로딩 + 예열) |
| 지능 (공개 평가) | Artificial Analysis 지수 43 (추론 모드). Claude Haiku 5.5 같은 클라우드 소형 모델(36~43)과 비슷하고, 최상위 모델(58~59)보다는 확실히 아래 |

**"일반 AI 서비스와 비슷한가?"에 대한 솔직한 답**
- 속도와 사용감은 **비슷한 수준까지 왔어요.**
- 지식과 정확도는 **클라우드 소형 모델 수준**이에요. 최신 정보와 인터넷 검색이 없고, 사실 오류도 있었어요.
  예: "동의보감은 세종 때 편찬"이라고 답했는데, 실제로는 선조 때 허준의 업적이에요.

## 1. 웹 조사로 후보 좁히기 (다운로드 전)

| 모델 | AA 지수 | MMLU-Pro | 판단 |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | **43** | 85.2 | 선택. RTX 3070 8GB 노트북에서 약 33토큰/초라는 공개 보고가 있음 |
| Gemma 4 26B-A4B | 26~31 | 82.6 | 다국어(MMMLU 86.3)는 강하지만 종합 지능이 낮음 |
| GLM-4.7-Flash | 30 | — | 1차 연구에서 사용 |
| gpt-oss-20b | 공개 점수 없음 | — | 제외 |

### 1-1. 최신 모델 재조사 (2026-11-22, 다운로드 없이 웹 자료만)

※ 위 표의 "AA 지수 43"은 예전 지수(v4) 기준이에요. 현재 v5 지수는 척도가 달라요(오픈 모델 1위가 46, Qwen3.8-27B가 20). 그래서 두 버전의 점수를 직접 비교하면 안 돼요.

| 모델 | 크기 | 공개 평가 (v5) | 이 PC 급(8GB+16GB) 속도 | 판단 |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B (현재)** | 35B, 활성 3B | v5 점수 없음 | **35~37 tok/s (실측)** | 유지 |
| Inkling 2 Mini (11/20 출시) | 26B, 활성 3.4B, 반복 구조라 실효 5.4B | 20 | 약 17 tok/s (RTX 4060 8GB+16GB 사용자 보고) | 코딩·에이전트는 앞서지만, 속도가 절반이고 사실 정확도가 낮음(환각 없는 비율 63% vs 71%). 출시 2일차라 지원 엔진도 불안정함 |
| Qwen3.8-27B (8월) | 27B 전부 활성(dense) | 20 | 추정 5~8 tok/s (전체 가중치를 매 토큰 읽음) | 너무 느려서 제외 |
| Qwen3.8-Flash-Next | 125B+51B, 활성 6B | 29 | 파일이 40GB 이상이라 메모리 초과 | 제외 |
| Qwen3.8-35B-A3B | — | — | — | **공식 미출시**(11/22 Hugging Face Qwen 계정 전수 확인: 3.8은 27B·Flash-Next·2.4T만 있음). 같은 이름의 커뮤니티 모델(empero-ai Distill, Whittle 등)은 Qwen3.6-35B-A3B를 개조한 비공식판. 제작자 자체 측정에서도 지식은 같거나 낮음(MMLU 0.838→0.834, 희귀 사실 132→94~111) → 교체 이득 없음 |
| Nemotron 3.5 Lightning 30B-A3B | 30B, 활성 3B | v5 점수 불확실 | — | NVIDIA 모델 카드에서도 Qwen3.6-35B-A3B에 14개 중 12개 열세. Q3_K_M도 19.8GB라 너무 큼 |

결론: 지금 이 PC에서는 **Qwen3.6-35B-A3B 유지**가 최선이에요. 지금 가장 아쉬운 점이 사실 오류인데, Inkling 2 Mini는 이 항목이 더 나빠요. Qwen3.8-35B-A3B가 나오면 그때 3비트 파일 하나만 받아 시험하면 돼요.
출처: https://artificialanalysis.ai/leaderboards/models?weights=open · https://huggingface.co/thinkingmachines/Inkling-2-Mini · https://benchlm.ai/compare/inkling-2-mini-vs-qwen3-6-35b-a3b · https://x.com/analogalok/status/2124135141555720653 · https://the-decoder.com/alibabas-qwen-team-releases-qwen-3-8-models-with-open-weights-under-the-apache-2-0-license/

## 2. 이 PC에서만 알 수 있었던 것 (직접 측정)

1. **4비트 파일(17GB)은 RAM 16GB에 다 들어가지 않아요.** 생성 속도 24토큰/초, 질문 읽기 초당 40~370토큰에 그쳤어요.
   → **3비트(IQ3_S, 12.7GB)로 바꾸자 35~37토큰/초**가 나왔어요.
   품질 손실은 Unsloth의 같은 계열 측정치로 판단했어요. 평균 오차(KLD)가 4비트 0.024 대비 3비트급 0.03~0.05로 작고, Unsloth도 이 정도를 권장 절충점으로 봐요.
2. **MTP(미리 예측해서 속도를 올리는 기능)는 이 PC에서 손해**였어요(24.5 → 21.5토큰/초). 전문가를 CPU에 둔 구성에서는 검증할 토큰이 늘수록 CPU 계산이 늘기 때문이에요.
3. **`-lm none`(모델을 통째로 RAM에 올리는 방식)은 빠르지만 위험해서 쓰지 않았어요.** 질문 읽기는 초당 1,000토큰까지 나왔지만, 시스템 메모리 예약이 한도(23.2GB)를 넘는 29.9GB가 돼서 Windows가 페이지 파일을 자동으로 늘렸어요. 다른 프로그램이 메모리 부족을 겪을 수 있어요.
4. **첫 질문이 느린 원인**: 서버를 켠 직후에는 전문가 가중치를 SSD에서 처음 읽어요. 그런데 **언어마다 켜지는 전문가가 달라서**(1차 연구에서 확인), 영어로만 예열하면 한국어 첫 질문이 여전히 느렸어요.
   → 영어, 한국어, 코드 세 가지로 예열하도록 바꾸자 첫 질문도 1.3초가 됐어요.
5. **대화 칸을 1개로 줄임(`-np 1`)**: 혼자 쓰는 PC라 칸이 4개일 필요가 없어요. 1개로 줄이자 이어지는 질문은 앞 대화를 재사용해서 0.4초 만에 읽기 시작했어요.
6. 최적 배치: 40층 중 26층의 전문가를 RAM에 두고, 나머지와 공통 부분은 GPU에 둬요(VRAM 7.3GB 사용). 질문은 1024토큰 단위로 읽고, CPU 스레드는 6개를 써요.

## 3. 안전 조치

- **설치 프로그램 없음.** 관리자 권한, 레지스트리, 시스템 설정은 건드리지 않았어요.
- 이 PC 안(127.0.0.1)에서만 접속돼요. 같은 와이파이의 다른 기기에서도 접속할 수 없어요.
- 모든 다운로드는 공식 출처이고, SHA256 해시로 검증했어요.
- 만든 것은 `<local>` 폴더, `<local>` 폴더, 바탕화면 바로가기 2개뿐이에요. 지우는 방법은 사용법.md에 있어요.
- 다 쓰면 「내 AI 종료」로 GPU와 RAM을 즉시 돌려받을 수 있어요(확인함: RAM 11.2GB 여유, VRAM 0.4GB 사용으로 복귀).

## 4. 실패와 정정 기록

- 처음에 후보 모델 47GB를 한꺼번에 받으려다 지적을 받고 멈췄어요. 이후 웹 조사로 1개만 골라 받았어요.
  3비트 파일은 측정 결과로 필요성이 확인된 뒤에 받았어요.
- 제 로컬 정확도 평가 도구에 결함이 있었어요(답변 길이를 16토큰으로 잘라 정답 추출이 틀림). 그래서 GLM 점수(30%)는 무효 처리했어요.
- 소규모 점검(생각 모드, 문항 6개씩)에서 수학 6/6, 영어 전문지식 3/6, 한국어 전문지식 2/6을 맞혔어요. 표본이 작아 통계적 의미는 없지만, **한국 고유의 전문지식이 약할 수 있다는 신호**로 기록해 둬요.

## 5. 출처
- Qwen3.6 모델 카드(미러): https://huggingface.co/Simplismart/Qwen3.6-35B-A3B
- Gemma 4 모델 카드: https://ai.google.dev/gemma/docs/core/model_card_4
- Artificial Analysis: https://artificialanalysis.ai/models/qwen3-6-35b-a3b · https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5
- RTX 3070 8GB 실측 보고: https://www.xda-developers.com/i-stopped-thinking-my-8gb-gpu-was-too-small-after-learning-to-offload-moe-llm-layers-to-ram/ · https://note.com/ripe_hosta6920/n/nb81b72d54ad7?hl=en
- llama.cpp MTP: https://github.com/ggml-org/llama.cpp/pull/22673
- Unsloth 양자화 벤치마크: https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
