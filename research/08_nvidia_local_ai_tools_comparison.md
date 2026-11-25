# 엔비디아 공식 도구, 지금 구성을 넘지 못한다

작성: 2026-11-22 · 대상 PC: RTX 3070 8GB(Ampere) / RAM 16GB / Ryzen 5 5600X / Windows 11 · 비교 기준: llama.cpp b12606(CUDA) 내장 웹 UI + Qwen3.6-35B-A3B UD-IQ3_S(12.7GB), 실측 생성 35~37 tok/s, 첫 답변까지 약 1.3초, 질문 읽기 약 360 tok/s

**결론부터 말하면, 2026년 11월 기준 엔비디아가 직접 내놓은 로컬 AI 도구 가운데 이 PC에서 지금 구성보다 속도, 정확도, 지식을 확실히 올려 주는 것은 없습니다.** 앱 쪽을 보면 ChatRTX는 2026년 1월에 개발이 끝났고, G-Assist는 대화형 AI가 아니라 PC 조작 도우미입니다. AI Workbench는 WSL2(Windows 안에서 리눅스를 돌리는 기능)와 Docker를 설치하는 개발자 도구입니다. 엔진 쪽(TensorRT-LLM, TensorRT for RTX, Edge-LLM, NIM, Dynamo)은 모두 모델 전체를 그래픽카드 메모리(VRAM)에 올려야 합니다. 그런데 4비트로 줄여도 18~20GB인 35B 모델은 8GB에 들어가지 않습니다. 지금 속도의 비결인 "전문가(MoE 모델의 부분 신경망) 일부를 CPU에서 계산하는" 기능도 이 엔진들에는 없습니다. 엔비디아 자체 모델인 Nemotron 가운데 이 PC에 들어가는 크기는 Qwen3.6-35B-A3B보다 약하고, 더 똑똑한 Bolt 74B는 너무 큽니다. 엔비디아가 2026년에 공개한 속도 개선은 대부분 llama.cpp 본체에 직접 들어갔고, 지금 쓰는 b12606에 이미 포함돼 있습니다. 그래서 남은 일은 시스템을 건드리지 않는 실행 옵션 몇 가지를 선택적으로 시험해 보는 것뿐입니다. 드라이버 설정은 속도를 올려 주지 않습니다.

## 엔비디아 앱 세 가지는 끝났거나, 용도가 다르거나, 너무 무겁다

**ChatRTX는 2026년 1월 21일 공식 종료됐습니다.** GitHub 저장소 안내문에 "더 이상 유지보수하지 않는다"고 적혀 있고, 저장소는 읽기 전용으로 보관 처리됐습니다 ([GitHub NVIDIA/ChatRTX](https://github.com/NVIDIA/ChatRTX)). 개발자 포럼도 같은 공지와 함께 잠겼고, 후속 제품은 따로 안내하지 않았습니다 ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/chatrtx-has-been-deprecated-this-forum-is-locked/365743)). 종료되기 전에도 RTX 30 시리즈 8GB 카드에서 고를 수 있는 모델은 Mistral 7B, ChatGLM3 6B 같은 7B급뿐이었습니다. Llama 3.1 8B NIM은 RTX 4080 이상에서만 지원됐습니다 ([GitHub NVIDIA/ChatRTX](https://github.com/NVIDIA/ChatRTX)). 내 문서를 읽고 답하는 RAG 기능이 장점이었지만, 이제는 보안 업데이트가 없는 데다 설치할 때 외부 오픈소스 패키지까지 추가로 내려받습니다. 지금 쓰기에는 안전 면에서 손해입니다.

**G-Assist는 이 PC에서 돌아가기는 하지만 채팅 AI가 아닙니다.** 공식 페이지는 6GB 이상 VRAM과 드라이버 580.97 이상을 요구하고, 스스로 "폭넓은 대화형 AI가 아니다"라고 밝힙니다. 게임 설정 최적화, 오버클럭 적용, 하드웨어 상태 그래프, 주변기기 제어가 주 기능이고, 영어만 지원합니다 ([NVIDIA G-Assist](https://www.nvidia.com/en-us/software/nvidia-app/g-assist/)). 실행 중에는 VRAM이 4.5~6GB 비어 있어야 해서, 7.3GB를 쓰는 지금의 llama.cpp와 동시에 켤 수 없습니다. 안에 든 모델이 Llama 8B인지 Qwen 4B인지는 자료마다 다릅니다 ([Jon Peddie Research](https://www.jonpeddie.com/news/rtx-goes-local-cloud-and-cinematic-as-nvidia-deepens-its-ai-stack/)). 어느 쪽이든 35B급인 지금 모델보다 훨씬 작습니다. 오버클럭 기능이 있다는 점도 "PC 손상 금지" 원칙과 맞지 않습니다.

**AI Workbench는 2026년 8월에도 새 버전이 나오는 현역 도구지만, 개발자용 컨테이너 관리 도구입니다.** Windows에서는 WSL2를 켜고 전용 리눅스 배포판을 만들며, Docker Desktop 설정을 확인하고 재부팅을 요구할 수 있습니다 ([AI Workbench 설치 문서](https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/windows.html); [릴리스 노트](https://docs.nvidia.com/ai-workbench/release-notes/2026/8/5)). 관리자 권한이 필요한 시스템 변경이고, Windows 11 25H2에서 설치가 실패했다는 사용자 보고도 있습니다 ([NVIDIA Forums](https://forums.developer.nvidia.com/t/nvidia-workbench-install-failed-on-download-of-wsl-distro/351756)). 2026년 CES와 Computex 발표에서도 ChatRTX를 대신할 엔비디아 자체 채팅 앱은 나오지 않았습니다. 엔비디아는 LM Studio, Ollama, AnythingLLM 같은 외부 앱과 llama.cpp 개선을 안내하는 쪽으로 방향을 바꿨습니다 ([NVIDIA Blog, Computex 2026](https://blogs.nvidia.com/blog/rtx-ai-garage-computex-spark-local-agents); [NVIDIA AI on RTX](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/)). 이 외부 앱들도 속은 llama.cpp라서, 최신 llama.cpp를 직접 쓰는 지금보다 빨라질 이유가 없습니다.

## 엔비디아 추론 엔진은 35B 모델을 8GB에 넣지 못한다

추론 엔진은 모델을 실제로 계산하는 프로그램입니다. 엔비디아 엔진들이 이 PC에서 막히는 이유는 세 가지입니다. **첫째, 모델 전체(와 여유분)가 VRAM에 들어가야 합니다.** **둘째, MoE 전문가를 시스템 RAM에 두고 CPU로 계산하는 기능(llama.cpp의 `--n-cpu-moe`)이 없습니다.** **셋째, 4비트보다 작게 줄이는 형식이 없습니다.** 지금 쓰는 IQ3_S(약 3.4비트)가 12.7GB인데, 같은 모델을 4비트(INT4)로 만들면 18~20GB입니다. 이 값은 매개변수 수로 계산한 추정입니다.

| 엔진 | Windows에서 쓰는 방법 | 이 PC에서의 문제 | 시스템 변경 | 판단 |
|---|---|---|---|---|
| TensorRT-LLM | 공식적으로 Linux 전용이라 WSL2가 필요함 ([지원 표](https://nvidia.github.io/TensorRT-LLM/legacy/reference/support-matrix.html)) | 모델 전체가 VRAM에 들어가야 함. Ampere는 FP8/FP4를 지원하지 않음 | 있음(WSL2, 관리자 권한, 재부팅) | 불가 |
| TensorRT for RTX / Windows ML | Windows에서 바로 됨. Windows 업데이트로 자동 배포됨 ([KB5103216](https://support.microsoft.com/en-us/servicing/os/windows/ai-components/2026/06/kb5103216-nvidia-tensorrt-rtx-execution-provider-update-version-2-2606-3-0)) | 주 대상이 이미지·음성 모델임. 넘치는 가중치는 PCIe로 매번 실어 나르는 "가중치 스트리밍"으로 처리하고, 이때 CUDA Graphs가 꺼짐 ([문서](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/weight-streaming.html)) | 없음 | 추정으로 5 tok/s 미만. 불가 |
| TensorRT Edge-LLM | Windows는 "개발자" 등급이라 직접 빌드해야 하고 설치 파일이 없음 ([지원 표](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.1/user_guide/getting_started/support-matrix.html)) | Qwen3.6-35B-A3B를 지원하지만 "모델 크기 + 2GB"의 VRAM이 필요함 ([빠른 시작](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/quick-start-guide.html)) | 없음(빌드 도구 설치는 필요) | 35B는 불가. 9B 이하 소형 모델만 가능 |
| NIM for RTX | WSL2 필수 | **RTX 40/50만 지원하고 RTX 30은 빠져 있음** ([NIM WSL2](https://docs.nvidia.com/nim/wsl2/latest/getting-started.html)) | 있음(관리자 권한, BIOS 가상화, 재부팅) | 불가 |
| Dynamo | 해당 없음 | 데이터센터에서 여러 서버를 묶어 운영하는 도구임. 이 판단은 배경지식이고, 이번 조사에서 공식 지원 표는 확인하지 못함 | — | 무관 |

TensorRT for RTX의 가중치 스트리밍은 그나마 "VRAM보다 큰 모델"을 돌리는 방법입니다. 하지만 이 방식은 어떤 전문가가 필요한지 구분하지 않고 넘친 가중치를 토큰마다 PCIe로 보내는 것으로 보입니다. 토큰마다 10GB 넘게 옮기면, PCIe 4.0의 실효 속도(약 25GB/s)로는 초당 몇 토큰도 나오기 어렵습니다(계산에 따른 추정). llama.cpp는 이런 전송 자체를 피합니다. 토큰마다 실제로 쓰이는 약 3B 분량의 전문가만 5600X가 RAM에서 직접 계산하기 때문입니다. 이 방식이 8GB 카드에서 35 tok/s를 내는 핵심입니다.

엔비디아 엔진이 llama.cpp보다 빠르다는 비교 자료는 2024년 것입니다. Mistral 7B를 VRAM에 통째로 올린 조건에서 RTX 3090 기준 TensorRT-LLM이 144 tok/s, llama.cpp가 89 tok/s였습니다 ([Jan.ai](https://jan.ai/post/benchmarking-nvidia-tensorrt-llm)). 그 뒤 llama.cpp에 CUDA Graphs 등이 들어가면서 RTX 4090에서 약 150 tok/s까지 올라 격차가 크게 줄었습니다 ([NVIDIA Developer Blog](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)). 8GB 카드나 CPU 오프로드 조건의 비교는 찾지 못했습니다. 정리하면, 엔비디아 엔진이 이길 수 있는 경우는 9B 이하 작은 모델을 VRAM에 통째로 올릴 때뿐입니다. 그런 모델은 지금 모델보다 확실히 덜 똑똑합니다.

## Nemotron 모델은 Qwen3.6보다 약하거나 이 PC에 들어가지 않는다

엔비디아의 2026년 공개 모델 가운데 지금 모델과 같은 체급(전체 30B, 토큰마다 쓰는 부분 약 3B)은 Nemotron 3.5 Lightning, Cascade 2, Nemotron 3 Nano 30B입니다. **이 모델들은 같은 압축 수준에서 Qwen보다 파일이 큽니다.** Lightning은 3비트(Q3_K_M)가 19.8GB, 4비트가 25GB입니다 ([runaihome](https://runaihome.com/blog/nemotron-35-lightning-consumer-gpu-hardware-guide-2026/)). 지금 파일이 12.7GB이고, 4비트 17GB 파일만으로도 RAM 16GB가 넘쳐 속도가 24 tok/s로 떨어졌던 이 PC에서는 감당할 수 없는 크기입니다. 성능도 뒤집니다. 엔비디아가 직접 만든 Lightning 모델 카드의 비교표에서 **Qwen3.6-35B-A3B가 14개 시험 중 12개에서 이겼고**, Lightning은 지시 따르기 시험(IFBench, 71.88 대 63.71)에서만 앞섰습니다. 박사급 과학 문제(GPQA-D)는 75.44 대 83.40, 코딩(SWE-bench Verified)은 51.56 대 70.12였습니다 ([HF Lightning 카드](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)). 같은 카드는 Lightning의 지원 언어로 영어, 스페인어, 프랑스어, 독일어, 이탈리아어, 일본어만 적었고 **한국어는 목록에 없습니다.**

| 모델 | 크기(전체/활성) | 이 PC 적합성 | Qwen3.6-35B-A3B 대비 |
|---|---|---|---|
| Nemotron 3.5 Lightning 30B-A3B | 30B / 3B | 3비트도 19.8GB라 들어가지 않음 | 14개 중 12개 열세, 한국어 미지원 |
| Nemotron Cascade 2 30B-A3B | 30B / 3B | 3비트 약 19GB ([bartowski](https://huggingface.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF)) | AA 지수(v4) 28 대 43 ([Artificial Analysis](https://x.com/ArtificialAnlys/status/2049881951260283097?lang=en)) |
| Nemotron 3.5 Bolt 74B-A9B 24GB | 74B / 9B | 가중치 약 22GB, 공식 권장은 32GB 이상 GPU, Ada/Blackwell만 지원 ([HF Bolt 카드](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)) | 대부분 앞서지만 이 PC에서는 실행 불가 |
| Nemotron 3.5 Bolt E4B | 약 4B | 6.18GB라 VRAM에 통째로 들어감 ([Unsloth](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF)) | 4B급이라 체급 자체가 낮음 |
| Nemotron Nano 9B v2 / Nano 4B | 9B, 4B | VRAM에 들어감 | 체급이 낮음 ([HF 카드](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)) |

"정확도" 면에서 눈여겨볼 수치는 하나 있습니다. 모르는 것을 지어내지 않는 비율(AA-Omniscience non-hallucination)에서 Lightning이 68.81로 Qwen의 65.85보다 약간 높습니다. 하지만 아는 것을 맞히는 비율(정확도)은 16.63 대 19.47로 오히려 낮습니다 ([HF Bolt 카드](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)). 사실을 더 많이 알아서가 아니라 "모른다"고 더 자주 답하는 차이로 보입니다. 이전 실측에서 "동의보감은 세종 때 편찬"처럼 한국 고유 지식이 약하다는 신호가 있었는데, 한국어를 지원하지 않는 Nemotron은 이 약점을 메우지 못합니다. 덧붙여, Lightning이 긴 대화에서도 속도가 떨어지지 않는다는 RTX 5090 측정이 있습니다(Mamba-2 구조 덕분). 다만 엔비디아가 발표한 빠른 속도는 llama.cpp가 아직 쓰지 못하는 보조 모델(DFlash) 기준입니다 ([runaihome](https://runaihome.com/blog/nemotron-35-lightning-consumer-gpu-hardware-guide-2026/)). 따라서 이 PC에 그대로 적용되지 않습니다.

## 엔비디아의 llama.cpp 개선은 이미 들어와 있고, 남은 시험은 선택 사항이다

**엔비디아가 2026년에 공개한 로컬 AI 속도 개선은 대부분 llama.cpp 본체에 직접 들어갔습니다.** CES 2026 발표에서 엔비디아는 MoE 모델의 생성 속도가 4개월 동안 약 35% 빨라졌다고 했습니다. 내용은 행렬 계산 커널 개선, GPU에서 다음 단어 고르기(샘플링), 모델 로딩 15% 단축 등입니다. FP4 형식 가속은 RTX 50(Blackwell)에서만 됩니다 ([NVIDIA Developer Blog, CES 2026](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs)). 지금 쓰는 b12606은 2026년 11월 21일 빌드라서 이 개선이 모두 포함돼 있습니다 ([릴리스 b12606](https://github.com/ggml-org/llama.cpp/releases/tag/b12606)). 즉 **지금 구성이 이미 엔비디아가 권하는 길 위에 있습니다.**

엔비디아가 홍보한 MTP 예측 디코딩(여러 토큰을 미리 짐작해 한 번에 확인하는 기능)도 마찬가지입니다. Qwen3.6-35B에서 최대 1.6배라는 수치는 모델이 VRAM에 통째로 들어가는 RTX 5090에서 잰 것입니다 ([NVIDIA Blog, Computex 2026](https://blogs.nvidia.com/blog/rtx-ai-garage-computex-spark-local-agents)). 이 PC에서는 이미 직접 재 봤고, **24.5에서 21.5 tok/s로 오히려 느려졌습니다**(IQ4_XS 파일 기준). 전문가를 CPU가 계산하는 구성에서는 확인할 토큰이 늘수록 CPU 일이 늘기 때문입니다. 모델을 RAM에 통째로 올리는 `-lm none`도 이미 시험했습니다. 시스템 메모리 예약(commit charge)이 29.9GB로 한도 23.2GB를 넘어서 쓰지 않기로 했습니다.

GPU 전문가 캐시 옵션 `--moe-cache-mib`는 RAM에 있는 전문가 중 자주 쓰는 것을 GPU에 복사해 두는 기능입니다. 2026년 10월에 정식으로 들어갔고, 개발자는 RTX 4090/5090에서 1.6~2.2배를 보고했습니다. 반면 RTX 3090에서는 잘 맞춘 `--n-cpu-moe` 대비 1~5% 향상에 그쳤다는 보고가 있습니다 ([GitHub PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887); [developersdigest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)). 8GB 카드에서 같은 Qwen3.6-35B-A3B로 1.72배를 냈다는 기록도 있지만, 이것은 정식 채택되지 않고 닫힌 다른 구현입니다 ([GitHub PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563)). 무엇보다 **이 PC의 1차 실험(GLM-4.7-Flash)에서 이 캐시를 켜자 8.2~9.0에서 5.7~6.5 tok/s로 오히려 느려졌습니다.** 캐시에 없는 전문가를 그때그때 PCIe로 올리는 비용이 컸기 때문입니다. 8GB에서는 캐시에 VRAM을 주는 만큼 GPU에 고정해 둘 전문가 층을 빼야 한다는 문제도 있습니다. Windows에서는 mmap 방식(파일을 필요할 때 읽는 방식)과 함께 쓸 때 추가 비용이 생긴다는 보고도 있습니다 ([GitHub PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)). Qwen3.6 IQ3_S로는 아직 재 보지 않았으니 다시 시험할 수는 있지만, **우선순위는 낮고 효과도 보장되지 않습니다.**

아래 표는 남은 선택지를 위험도와 시스템 변경 여부로 정리한 것입니다. 1~5번은 모두 실행할 때 붙이는 옵션이라, 지우기만 하면 원래대로 돌아갑니다.

| # | 변경 | 기대 효과 | 위험 | 시스템/드라이버 설정 변경 | 되돌리기 |
|---|---|---|---|---|---|
| 1 | `-ub 2048`(질문을 읽는 단위를 키움) | 긴 질문을 읽는 속도가 오를 수 있음. 대신 VRAM을 더 써서 전문가 층을 1개 빼야 할 수 있음 ([HF 오프로드 가이드](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)) | 매우 낮음(최악이면 메모리 부족으로 실행 실패) | 없음 | 옵션 삭제 |
| 2 | `--backend-sampling`(다음 단어 고르기를 GPU에서) | 엔비디아 측정 조건에 포함된 옵션. 효과는 작을 것으로 예상 ([NVIDIA CES 2026](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs)) | 매우 낮음 | 없음 | 옵션 삭제 |
| 3 | `GGML_CUDA_GRAPH_OPT=1`(실행 창 안에서만 쓰는 환경 변수) | 거의 없을 것으로 예상. CUDA Graphs가 켜져 있어야 작동하는데, `--n-cpu-moe`를 쓰면 CUDA Graphs가 꺼짐 ([PR #18496](https://github.com/ggml-org/llama.cpp/pull/18496); [PR #18593](https://github.com/ggml-org/llama.cpp/pull/18593)) | 매우 낮음(오류가 나도 해당 프로그램만 종료) | 없음(그 창에만 적용) | 창 닫기 |
| 4 | `--moe-cache-mib 256~1024` + `--n-cpu-moe` 27~30 | 불확실. 이 PC의 1차 실험에서는 느려졌음 | 매우 낮음 | 없음 | 옵션 삭제 |
| 5 | `-ot`로 전문가 층을 정확히 지정 | `--n-cpu-moe`와 같거나 VRAM을 약간 더 알뜰하게 씀 ([HF 오프로드 가이드](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)) | 매우 낮음 | 없음 | 옵션 삭제 |
| 6 | NVIDIA 제어판에서 `llama-server.exe`만 "CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback"으로 설정 | 속도는 그대로. VRAM이 넘칠 때 조용히 느려지는 대신 오류가 나서 원인을 바로 알 수 있음 ([runaihome](https://runaihome.com/blog/shared-gpu-memory-slow-local-ai-sysmem-fallback-fix-2026/)) | 낮음 | **있음**(드라이버의 프로그램별 설정. 사용자 동의 필요) | "전역 설정 사용"으로 되돌림 |
| 7 | HAGS(하드웨어 가속 GPU 예약) 켜기/끄기 | llama.cpp에서 효과가 있다는 근거 없음 ([Tom's Hardware](https://www.tomshardware.com/news/gpu-hardware-scheduling-tested)) | 낮음, 재부팅 필요 | **있음** | 가능. **권장하지 않음** |
| 8 | 전력 제한 상향, 오버클럭 | 대상 아님 | 중간 이상 | **있음** | **하지 말 것** |

드라이버 설정 중 이 PC에 의미가 있는 것은 6번뿐이고, 이것도 속도를 올리지는 않습니다. 지금처럼 35~37 tok/s가 꾸준히 나온다면 VRAM이 넘치고 있지 않다는 뜻이라 굳이 바꿀 필요가 없습니다. 1~5번을 시험하면서 VRAM 한계를 밀어붙일 때만 진단용으로 쓸 만합니다. llama.cpp를 업데이트할 때 주의할 점도 하나 있습니다. **CUDA 13.2로 빌드한 파일은 IQ3_S 모델에서 엉뚱한 글자를 출력하는 버그가 있었고, 13.3에서 고쳐졌습니다** ([GitHub issue #21255](https://github.com/ggml-org/llama.cpp/issues/21255)). 지금 빌드는 CUDA 13.5라 해당하지 않지만, 나중에 업데이트할 때는 파일 이름에 적힌 CUDA 버전을 확인하는 것이 좋습니다.

## 결론

이번 조사로 바뀐 점은 이렇습니다. "엔비디아 공식 도구 = 엔비디아 GPU에서 가장 빠른 길"이라는 생각은 이 PC에서는 맞지 않습니다. 엔비디아의 전용 엔진은 VRAM이 넉넉한 RTX 40/50과 데이터센터를 전제로 만들어졌습니다. 8GB 카드에 큰 MoE 모델을 나눠 싣는 방식은 오픈소스인 llama.cpp만 제대로 지원하고, 엔비디아도 개인 PC용 개선을 llama.cpp에 직접 넣는 쪽을 택했습니다. 그래서 속도는 지금 구성이 사실상 이 하드웨어의 상한에 가깝습니다. 남은 실행 옵션 시험의 기대 이득도 몇 % 수준입니다.

실제로 아쉬운 것은 속도가 아니라 지식과 사실 정확도, 특히 한국 고유 지식입니다. 이 문제는 엔비디아 제품군 안에서는 풀리지 않습니다. Nemotron은 한국어를 지원하지 않고, 내 문서 기반 답변(RAG)을 해 주던 ChatRTX는 종료됐습니다. 이 방향으로 개선하려면 엔비디아 바깥에서 찾아야 합니다. 이전 조사에서 1순위로 꼽은 Qwen3.8-35B-A3B 같은 더 나은 같은 체급 모델이 나오기를 기다리거나, 내 자료를 붙여 쓰는 외부 앱(엔비디아가 안내하는 AnythingLLM, LM Studio 등)을 별도로 검토하는 방법이 있습니다. 엔비디아 쪽에서 다시 볼 만한 계기는 하드웨어가 바뀔 때(VRAM 24GB 이상 카드)뿐입니다. 그때는 Bolt 74B와 TensorRT 계열 엔진을 다시 검토할 만합니다.
