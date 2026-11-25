# 연구 기록 모음

> 8GB 그래픽카드 한 장으로 35B 모델을 돌리고, 가르치고, 결국 학습까지 시켜 본 과정을 순서대로 적은 글들. (English summary at the bottom)

처음 목표는 소박했다. "이 컴퓨터에서 큰 모델이 돌아가기는 할까?" 돌아가니까 똑똑하게 만들고 싶어졌고, 똑똑해지니까 직접 가르치고 싶어졌다. 아래는 그 순서 그대로다. 최종 결과는 [기술 보고서](../REPORT.md)에 정리해 뒀다.

| 편 | 글 | 한 줄 요약 |
|---|---|---|
| 1 | [큰 MoE를 VRAM·RAM·SSD에 나눠 돌려 보기](01_moe_3tier_offload.md) | 전문가를 세 층으로 나눠 담으면 GPU보다 큰 모델도 돈다 |
| 2 | [이 컴퓨터에 제일 잘 맞는 로컬 AI 찾기](02_best_local_ai_for_8gb.md) | 받기 전에 먼저 조사하고, 모델·양자화·엔진을 비교해서 골랐다 |
| 3 | [한국어 지식 붙이기와 "해마 기억"](03_korean_knowledge_hippocampal_memory.md) | 로컬 검색으로 희귀 사실 정답률이 11%에서 72%로. 한 번 보고 기억하는 구조도 만들어 봤다 |
| 4 | [한 번 듣고 배우고, 짧게 생각하기](04_learn_once_think_efficiently.md) | "기억해"라고 한 번 말하면 새 대화에서도 기억한다. 3비트 모델은 오래 생각할수록 손해였다 |
| 5 | [모델 중간 층을 직접 학습시키기](05_mid_layer_weight_training.md) | 8GB로 실제 모델의 층을 학습하는 길을 뚫었다 |
| 6 | [시험 풀이 어댑터와 전문가 캐싱](06_exam_adapter_and_expert_cache.md) | 단계별 풀이 정답률 24%에서 52%로, 생성 속도는 초당 38에서 60토큰으로 |
| 7 | [더 빨라질 수 있을까? 추가 최적화 조사](07_further_local_moe_optimization.md) | 속도는 거의 끝까지 왔다. 남은 이득은 데이터와 추론 쪽에 있다 |
| 8 | [엔비디아 공식 도구를 써 보면 나을까?](08_nvidia_local_ai_tools_comparison.md) | 2026년 11월 기준, 지금 구성을 넘는 공식 도구는 없었다 |

글 안의 로컬 경로는 `<local>`로 바꿨다. 학습 데이터와 데이터를 모은 과정은 공개하지 않는다. 공개 벤치마크와 웹 자료는 각 글에 출처를 남겼다.

---

**English summary.** Eight posts (in Korean), in the order the work happened: running a 35B MoE across VRAM/RAM/SSD (1), choosing the best local setup (2), adding local Korean knowledge retrieval and a one-shot "hippocampal" memory (3–4), training real mid-layer weights on 8 GB (5), the exam adapter and expert caching (6), and two surveys on further optimization and NVIDIA's local tools (7–8). The final result is in [../REPORT.md](../REPORT.md).
