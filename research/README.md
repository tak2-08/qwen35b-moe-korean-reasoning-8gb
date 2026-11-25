# 연구 기록 / Research log

RTX 3070 8GB · RAM 16GB PC 한 대에서 35B MoE 모델(Qwen3.6-35B-A3B)을 돌리고, 지식을 붙이고, 학습시키기까지의 단계별 연구 보고서입니다 (한국어). 최종 결과는 상위 폴더의 [REPORT.md](../REPORT.md)에 있습니다.

Step-by-step research reports (Korean) from running, augmenting and fine-tuning a 35B MoE on one 8 GB GPU. The final result is in [../REPORT.md](../REPORT.md).

| # | 보고서 | 핵심 |
|---|---|---|
| 1 | [MoE 3단(VRAM-RAM-SSD) 실행](01_moe_3tier_offload.md) | 전문가를 VRAM·RAM·SSD에 나눠 큰 MoE 실행 |
| 2 | [이 PC에 맞는 로컬 AI 찾기](02_best_local_ai_for_8gb.md) | 모델·양자화·엔진 비교와 선택 |
| 3 | [한국어 지식 보강 + 해마 기억](03_korean_knowledge_hippocampal_memory.md) | 로컬 검색으로 희귀 사실 정답률 11% → 72%, 한 번 보고 기억하는 새 구조 |
| 4 | [한 번 듣고 배우기, 효율적으로 생각하기](04_learn_once_think_efficiently.md) | 빠른 기억 + 짧은 사고 정책 |
| 5 | [중간 층 가중치 학습](05_mid_layer_weight_training.md) | 8GB에서 실제 모델 층 학습 경로 |
| 6 | [시험 어댑터 + 전문가 캐싱](06_exam_adapter_and_expert_cache.md) | 단계별 풀이 24% → 52%, 생성 38 → 60 tok/s |
| 7 | [추가 최적화 조사](07_further_local_moe_optimization.md) | 남은 이득은 엔진보다 데이터·추론 쪽 |
| 8 | [엔비디아 로컬 AI 도구 비교](08_nvidia_local_ai_tools_comparison.md) | 공식 도구는 현재 구성을 넘지 못함 |

로컬 경로는 `<local>`로 바꿨고, 학습 데이터와 수집 과정은 공개하지 않습니다. 공개 벤치마크와 웹 자료는 각 보고서에 출처를 남겼습니다.
