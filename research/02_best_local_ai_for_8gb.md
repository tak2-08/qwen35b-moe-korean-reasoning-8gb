# 2편. 이 컴퓨터에 제일 잘 맞는 로컬 AI 찾기

> 1편에서 '돌릴 수는 있다'는 걸 확인했으니, 이번엔 '무엇을 돌릴까'를 골랐다. 큰 파일부터 받는 대신 공개된 벤치마크를 먼저 찾아보고, 후보를 좁힌 다음에만 받아서 쟀다.

2026-11-22 · RTX 3070 8GB / RAM 16GB / Ryzen 5 5600X

## 결론부터

**Qwen3.6-35B-A3B (UD-IQ3_S, 12.7GB) + llama.cpp 내장 채팅 화면**으로 정했다.

| 항목 | 실제 채팅 화면에서 잰 값 |
|---|---|
| 답변 생성 속도 | **초당 35~37토큰** |
| 질문을 보내고 답이 시작될 때까지 | **약 1.3초** |
| 켜는 데 걸리는 시간 | 약 35초 (모델 로딩 + 예열) |
| 지능 (공개 평가) | Artificial Analysis 지수 43 (추론 모드). 클라우드 소형 모델(36~43)과 비슷하고, 최상위 모델(58~59)보다는 확실히 아래 |

**"그래서 클라우드 AI랑 비슷하냐?"고 물으면**
- 속도와 쓰는 느낌은 꽤 비슷해졌다.
- 지식과 정확도는 클라우드 소형 모델 정도다. 최신 정보도, 인터넷 검색도 없고, 사실을 틀리기도 한다. 예를 들어 "동의보감은 세종 때 편찬"이라고 답했다. 실제로는 선조 때 허준이 썼다.

## 받기 전에 먼저 조사했다

| 모델 | AA 지수 | MMLU-Pro | 판단 |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | **43** | 85.2 | 선택. RTX 3070 8GB 노트북에서 약 33토큰/초라는 공개 보고가 있었다 |
| Gemma 4 26B-A4B | 26~31 | 82.6 | 다국어(MMMLU 86.3)는 강한데 종합 지능이 낮다 |
| GLM-4.7-Flash | 30 | — | 1편에서 썼던 모델 |
| gpt-oss-20b | 공개 점수 없음 | — | 제외 |

### 최신 모델도 다시 훑어봤다 (2026-11-22, 웹 자료만)

참고로 위 표의 "AA 지수 43"은 예전 지수(v4) 기준이다. 지금 v5 지수는 척도가 달라서(오픈 모델 1위가 46, Qwen3.8-27B가 20) 두 버전 점수를 바로 비교하면 안 된다.

| 모델 | 크기 | 공개 평가 (v5) | 이 정도 PC(8GB+16GB) 속도 | 판단 |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B (현재)** | 35B, 활성 3B | v5 점수 없음 | **35~37 tok/s (실측)** | 유지 |
| Inkling 2 Mini (11/20 출시) | 26B, 활성 3.4B, 반복 구조라 실효 5.4B | 20 | 약 17 tok/s (RTX 4060 8GB+16GB 사용자 보고) | 코딩·에이전트는 앞서지만 속도가 절반이고 사실 정확도가 낮다(환각 없는 비율 63% vs 71%). 출시 이틀째라 엔진 지원도 불안정 |
| Qwen3.8-27B (8월) | 27B 전부 활성(dense) | 20 | 추정 5~8 tok/s (매 토큰 전체 가중치를 읽음) | 너무 느려서 제외 |
| Qwen3.8-Flash-Next | 125B+51B, 활성 6B | 29 | 파일만 40GB 이상이라 메모리 초과 | 제외 |
| Qwen3.8-35B-A3B | — | — | — | **공식으로는 아직 없다.** Hugging Face의 Qwen 계정을 다 뒤져 봤는데 3.8은 27B·Flash-Next·2.4T뿐이었다. 같은 이름의 커뮤니티 모델(empero-ai Distill, Whittle 등)은 Qwen3.6-35B-A3B를 개조한 비공식판이고, 만든 사람이 직접 잰 수치로도 지식이 같거나 낮다(MMLU 0.838→0.834, 희귀 사실 132→94~111). 바꿀 이유가 없다 |
| Nemotron 3.5 Lightning 30B-A3B | 30B, 활성 3B | v5 점수 불확실 | — | NVIDIA 모델 카드에서도 Qwen3.6-35B-A3B에 14개 중 12개 항목에서 진다. Q3_K_M도 19.8GB라 너무 크다 |

그래서 **Qwen3.6-35B-A3B를 그대로 쓰기로 했다.** 지금 제일 아쉬운 게 사실 오류인데, Inkling 2 Mini는 그 부분이 더 나쁘다. Qwen3.8-35B-A3B가 정식으로 나오면 그때 3비트 파일 하나만 받아 보면 된다.

출처: https://artificialanalysis.ai/leaderboards/models?weights=open · https://huggingface.co/thinkingmachines/Inkling-2-Mini · https://benchlm.ai/compare/inkling-2-mini-vs-qwen3-6-35b-a3b · https://x.com/analogalok/status/2124135141555720653 · https://the-decoder.com/alibabas-qwen-team-releases-qwen-3-8-models-with-open-weights-under-the-apache-2-0-license/

## 직접 돌려 보고서야 알게 된 것

1. **4비트 파일(17GB)은 RAM 16GB에 다 안 들어간다.** 생성 24토큰/초, 질문 읽기는 초당 40~370토큰에 그쳤다. **3비트(IQ3_S, 12.7GB)로 바꾸자 35~37토큰/초**가 나왔다. 품질 손실은 Unsloth의 같은 계열 측정치를 보고 판단했다. 평균 오차(KLD)가 4비트 0.024, 3비트급 0.03~0.05로 작고, Unsloth도 이 정도를 권장 절충점으로 본다.
2. **MTP(미리 예측해서 빠르게 하는 기능)는 오히려 손해였다**(24.5 → 21.5토큰/초). 전문가를 CPU에 둔 구성에서는 검증할 토큰이 늘수록 CPU 계산도 같이 늘기 때문이다.
3. **`-lm none`(모델을 통째로 RAM에 올리기)은 빠르지만 쓰지 않았다.** 질문 읽기가 초당 1,000토큰까지 나왔지만, 메모리 예약이 한도(23.2GB)를 넘는 29.9GB가 되면서 Windows가 페이지 파일을 멋대로 늘렸다. 다른 프로그램이 메모리 부족에 걸릴 수 있다.
4. **첫 질문이 느렸던 이유.** 서버를 켠 직후에는 전문가 가중치를 SSD에서 처음 읽는다. 그런데 1편에서 봤듯이 **언어마다 켜지는 전문가가 다르다.** 영어로만 예열하니 한국어 첫 질문은 여전히 느렸다. 영어·한국어·코드 세 가지로 예열하게 바꾸니 첫 질문도 1.3초가 됐다.
5. **대화 칸을 하나로 줄였다(`-np 1`).** 혼자 쓰는 PC에 칸 4개는 필요 없다. 하나로 줄이니 이어지는 질문은 앞 대화를 재사용해서 0.4초 만에 읽기 시작했다.
6. **최적 배치.** 40층 중 26층의 전문가를 RAM에, 나머지와 공통 부분은 GPU에 뒀다(VRAM 7.3GB). 질문은 1024토큰 단위로 읽고, CPU 스레드는 6개.

## 지킨 원칙

- 설치 프로그램 없이 갔다. 관리자 권한, 레지스트리, 시스템 설정은 손대지 않았다.
- 이 컴퓨터 안(127.0.0.1)에서만 접속된다. 같은 와이파이의 다른 기기에서도 못 들어온다.
- 받은 파일은 전부 공식 출처였고, SHA256 해시로 확인했다.
- 다 쓰고 끄면 GPU와 RAM이 바로 돌아온다(확인: RAM 11.2GB 여유, VRAM 0.4GB로 복귀).

## 삽질 기록

- 처음엔 후보 모델 47GB를 한꺼번에 받으려고 했다. 생각해 보니 낭비라 멈추고, 웹 조사로 하나만 골라 받았다. 3비트 파일도 측정으로 필요하다는 게 확인된 뒤에 받았다.
- 직접 만든 정확도 평가 도구에 버그가 있었다. 답을 16토큰에서 잘라 버려서 정답 추출이 틀렸다. 그래서 GLM 점수(30%)는 버렸다.
- 생각 모드로 분야별 6문항씩 작게 점검해 보니 수학 6/6, 영어 전문지식 3/6, 한국어 전문지식 2/6이었다. 표본이 작아 통계적 의미는 없지만, **한국 고유 지식이 약할 수 있다는 신호**로 적어 둔다. 이게 3편의 출발점이 됐다.

## 참고한 자료
- Qwen3.6 모델 카드(미러): https://huggingface.co/Simplismart/Qwen3.6-35B-A3B
- Gemma 4 모델 카드: https://ai.google.dev/gemma/docs/core/model_card_4
- Artificial Analysis: https://artificialanalysis.ai/models/qwen3-6-35b-a3b · https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5
- RTX 3070 8GB 실측 보고: https://www.xda-developers.com/i-stopped-thinking-my-8gb-gpu-was-too-small-after-learning-to-offload-moe-llm-layers-to-ram/ · https://note.com/ripe_hosta6920/n/nb81b72d54ad7?hl=en
- llama.cpp MTP: https://github.com/ggml-org/llama.cpp/pull/22673
- Unsloth 양자화 벤치마크: https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks

---

[← 연구 기록 모음으로](README.md)
