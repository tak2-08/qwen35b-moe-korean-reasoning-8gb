# I fine-tuned a 35B model on an 8 GB GPU

**English** | [한국어](README.ko.md)

> A personal research log: adding a LoRA to Qwen3.6-35B-A3B (MoE) so it solves Korean exam questions step by step, better and shorter.

## How it started

My PC is ordinary: RTX 3070 8 GB, 16 GB RAM, Ryzen 5 5600X. Plenty of people *run* 35B models on hardware like this. I wanted to know the next thing: **can this PC actually train one?**

Not the usual way. Even at 3 bits the model is 12.7 GB, so it never fits on the GPU, and standard training stacks can't even load it. So I looked for a way around.

## Results first

Measured on the newest exam year, which was never used for training.

| | Base | My adapter |
|---|---|---|
| Step-by-step accuracy (220 items, ≤ 2000 tokens) | 53.2% | **63.6%** (paired p = 0.006) |
| Average answer length | about 1,330 tokens | **about 540 tokens** |
| Answers cut off at 2000 tokens | 30.5% | **5.0%** |
| Direct answer, no reasoning (1,416 items) | **62.8%** | 60.5% |
| Korean benchmark mix (KMMLU, HAE-RAE, CSAT Korean 2025, civil service; 925 items) | **62.7%** | 61.6% |
| Same top-1 next token as the base model on held-out Wikipedia text | 100% | 92.4% |

In one line: **more right answers in less than half the length.** It isn't free. Direct answers got a little worse, and general knowledge dropped about 1 point. So my chat app turns the adapter on only for multiple-choice exam questions.

## How it fits in 8 GB

**1. Record the experts, don't train them.** Most of an MoE's weight sits in its routed experts. Instead of loading them onto the GPU, I let the real engine (llama.cpp) run as usual and record every expert output. A PyTorch rewrite of the rest (attention, Gated DeltaNet, norms, shared expert) plugs those outputs in as constants. The 12 GB of expert weights never touches the GPU, and the gradients are still exact.

**2. Split the work across GPU, RAM and SSD.** The recorded activations are about 125 KB per token, roughly 70 GB per 460k-token chunk. Each chunk goes to SSD, gets trained for one epoch, then gets deleted.

**3. A 40× speedup from one fix.** Computing Gated DeltaNet token by token made one epoch take 3.6 hours. A 64-token chunked form (WY/UT transform) brought it to 7 minutes, matching the reference within 1e-6.

**4. Keep it from forgetting.** A strong KL penalty (weight 12) keeps the model close to the original on ordinary text. Every checkpoint must also pass a gate: at least 92% top-1 next-token agreement with the base model on Wikipedia. The final checkpoint fell to 91.3% and was rejected. I kept the one after 3 of 5 chunks.

## The road here

Training wasn't the plan at first. It started with just getting a big model to run on this PC, then adding knowledge, then making it remember things from one mention, and finally touching the weights. I wrote it up as 8 posts (in Korean): **→ [research log](research/README.md)**

The full technical report (method, every measurement, limitations) is in **[REPORT.md](REPORT.md)**.

## What's here and what isn't

- `src/`: training and evaluation pipeline (code map in REPORT.md §7)
- `REPORT.md`: technical report
- `research/`: 8 step-by-step research posts
- **Not included:** model weights, the adapter, training data, and how the data was collected
  - Training data: about 2,900 private Korean step-by-step solutions made with several teacher LLMs, filtered by automatic answer matching and repetition-loop filtering.

## Running it yourself

- llama.cpp (Windows CUDA build) for `llama-server`, and to build `src/hm_host.cpp` against `llama.h`
- Python 3.11+, PyTorch with CUDA, `numpy`, `gguf`, `requests`
- Paths come from environment variables: `MODEL`, `ENGINE`, `HM_DATA`, `EXAM_DATA`, `RESULTS`, `LAB`

## License

CC BY-NC 4.0: non-commercial research and education only. See [LICENSE](LICENSE). The base model and llama.cpp keep their own licenses.
