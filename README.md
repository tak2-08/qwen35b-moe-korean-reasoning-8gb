# Qwen3.6-35B-A3B Korean Reasoning on 8GB

**English** | [한국어](README.ko.md)

Fine-tuning a 35B Mixture-of-Experts model (Qwen3.6-35B-A3B, IQ3_S) for Korean step-by-step exam reasoning on a single consumer PC: RTX 3070 8 GB, 16 GB RAM, Ryzen 5 5600X.

8GB GPU 한 장과 RAM 16GB로 35B MoE 모델에 LoRA를 학습해, 한국어 객관식 시험 문제의 단계별 풀이 능력을 높인 기록입니다.

**Full technical report (Korean, with English summary): [REPORT.md](REPORT.md)**

## Key results (held-out newest exam year, never trained)

| | Base | Adopted adapter |
|---|---|---|
| Step-by-step accuracy (120 items, ≤ 2000 tokens) | 52.5% | **63.3%** (paired p = 0.041) |
| Average answer length | 1,388 tokens | **524 tokens** |
| Direct answer, no reasoning (1,416 items) | **62.8%** | 60.5% |
| Korean benchmark mix (KMMLU, HAE-RAE, CSAT Korean 2025, civil service; 925 items) | **62.7%** | 61.6% |
| Same top-1 next token as the original model on held-out Wikipedia text | 100% | 92.4% |

The adapter helps when the model solves step by step, and costs a little when it must answer at once. The chat app therefore enables it only for multiple-choice exam questions.

## How it fits in 8 GB

- **Frozen-routed-expert surrogate**
  - The real llama.cpp engine records every routed-expert output.
  - A PyTorch re-implementation of the remaining layers (attention, Gated DeltaNet, norms, shared expert) uses those outputs as constants.
  - This gives exact gradients through all layers above the edit. The 12 GB of expert weights never enters GPU memory.
- **GPU / RAM / SSD tiers**
  - Activations are captured to SSD in chunks of about 460k tokens, about 70 GB each.
  - Each chunk is trained for one epoch, then the capture is deleted.
- **Chunked Gated DeltaNet**
  - A 64-token WY/UT-transform form of the gated delta rule matches the reference within 1e-6.
  - It made training about 40× faster.
- **Forgetting control**
  - KL replay on background text (weight 12).
  - A hard real-engine gate of at least 92% next-token agreement.
  - The checkpoint after 3 of 5 chunks was adopted. The final one fell to 91.3% and was rejected.

## Repository contents

- `src/`: training and evaluation pipeline (see the code map in REPORT.md §7).
- `REPORT.md`: method, all measurements, limitations.
- Not included: model weights, the adapter, training data, and the data-collection process.
  - Training data: about 2,900 private Korean step-by-step (CoT) solutions, synthesized with multiple teacher LLMs and filtered by automatic answer matching and repetition-loop filtering.

## Requirements

- llama.cpp (Windows CUDA build) for `llama-server` and for building `src/hm_host.cpp` against `llama.h`.
- Python 3.11+, PyTorch with CUDA, `numpy`, `gguf`, `requests`.
- Paths are set with environment variables: `MODEL`, `ENGINE`, `HM_DATA`, `EXAM_DATA`, `RESULTS`, `LAB`.

## License

CC BY-NC 4.0: non-commercial research and educational use only. See [LICENSE](LICENSE). The base model and llama.cpp keep their own licenses.
