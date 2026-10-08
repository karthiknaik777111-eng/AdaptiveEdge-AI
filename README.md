# AdaptiveEdge — On-Device Conversational AI
> **"Local AI that dynamically adapts to the limits and workload of the edge device."**  
> **Hacknex 2026 — Track: On-Device Conversational Stack (`HNX26EPS08`)**  
> **100% On-Device • CPU-Only • Offline Runtime • Zero Cloud Inference**

---

## 1. Problem Statement

Standard conversational AI pipelines rely on cloud APIs (OpenAI, Claude, Gemini, ElevenLabs) that introduce:
1. **Severe privacy vulnerabilities** (sensitive user audio streamed to remote third-party servers).
2. **Mandatory network connectivity** (failing completely offline or in low-connectivity edge environments).
3. **High recurring cloud API latency and costs**.

However, running conversational AI entirely **on-device on consumer CPU hardware** faces severe bottlenecks:
- Monolithic LLMs saturate CPU cores, causing thermal throttling, system unresponsiveness, and audio stuttering.
- Fixed thread allocations fail to account for dynamic system workloads (e.g., background tasks, browser tabs).
- Sequential speech-to-text, LLM generation, and text-to-speech pipelines produce unacceptably high **Time To First Audio (TTFA)** without intelligent routing and streaming synthesis.

---

## 2. Core Innovation: Dynamic CPU Thread/Core Adaptation

The primary breakthrough of **AdaptiveEdge** is **closed-loop dynamic resource adaptation**. Rather than pinning inference to a static number of worker threads or overwhelming the host machine, the system continuously monitors host telemetry via an active controller and dynamically adjusts CPU thread allocation and inference parameters:

| Operating Mode | Host Load Trigger | AI CPU Threads | Available Logical Cores | Cache Sensitivity | LLM Max Tokens | Context Window | Behavior |
|---|---|---|---|---|---|---|---|
| **Performance** | System CPU < 50%, RAM < 82% | **8 Threads** | 16 Cores | Strict (0.65) | 80 tokens | 1024 tokens | Peak speed for local inference when host is idle |
| **Balanced** | System CPU 50–70%, RAM 82–88% | **4 Threads** | 16 Cores | Standard (0.44) | 40 tokens | 512 tokens | Balanced throughput, avoiding host contention |
| **Resource-Safe** | System CPU 70–85%, RAM 88–93% | **2 Threads** | 16 Cores | Aggressive (0.38) | 24 tokens | 256 tokens | Drops AI threads to preserve computer responsiveness |
| **Survival** | System CPU > 85%, RAM > 93% | **2 Threads** | 16 Cores | Strict Cache Preferred | 18 tokens | 128 tokens | Minimal token generation; guarantees zero pipeline crash |

### Dynamic Thread Scaling at Runtime
- **LLM Thread Control**: `Llama.set_n_threads(n_threads)` modifies the quantized GGUF execution threads dynamically per turn without reloading weights from disk.
- **ASR Thread Control**: `whisper.cpp` / `faster-whisper` transcription threads scale dynamically per audio frame.
- **Graceful Degradation**: Response length and token consumption scale down under heavy pressure before any system instability occurs.

---

## 3. Supporting Innovations

1. **Tiered Semantic Local Routing**:
   - Queries pass through a hierarchical 3-level local resolver:
     - **Level 1A**: Normalized exact string match (<0.02 ms)
     - **Level 1B**: Dynamic regex patterns for temporal queries (<0.05 ms)
     - **Level 1C**: Local dense vector cosine similarity via `all-MiniLM-L6-v2` (~5–15 ms on CPU)
   - Cache hits completely bypass the LLM, eliminating 100% of LLM compute and yielding responses in <150 ms.

2. **Multilingual Local ASR**:
   - Powered by `whisper.cpp` (GGML tiny) and fallback `faster-whisper` (CTranslate2 int8 CPU).
   - Real-time automatic language identification and local transcription across **English, Spanish, Hindi, German, and French** on CPU.

3. **Streaming & Ultra-Low Time To First Audio (TTFA)**:
   - Evaluates LLM output stream clause-by-clause, feeding first complete phrases into the local TTS engine immediately.
   - Reduces TTFA from seconds down to ~400–1000 ms on CPU cache misses, and <150 ms on cache hits.

4. **100% Offline & CPU-Only**:
   - Zero external API dependencies at runtime.
   - Verified zero outbound cloud AI network calls.
   - SAPI5 / pyttsx3 native 16-bit PCM WAV synthesis directly rendered to browser Web Audio and system speaker.

---

## 4. End-to-End Architecture

```
                    MICROPHONE INPUT (sounddevice / Web Audio)
                                       ↓
                             VOICE ACTIVITY DETECTION
                           (Local Silero VAD on CPU)
                                       ↓
                           MULTILINGUAL LOCAL ASR
                   (whisper.cpp / faster-whisper int8 CPU)
                                       ↓
                       TIERED SEMANTIC LOCAL ROUTER
                                       ↓
          ┌────────────────────────────────────────────────────────┐
          │                                                        │
     [CACHE HIT]                                              [CACHE MISS]
  Semantic Similarity ≥ Threshold                         Semantic Similarity < Threshold
  Exact / Rules / Dense Vector                            Local Quantized LLM
  Latency: ~10 – 30 ms                                    (Qwen2.5-0.5B-Instruct-Q4_K_M)
  LLM Bypassed: 0 ms Compute                              Streaming Clause Generator
          │                                                        │
          └───────────────────────────┬────────────────────────────┘
                                      ↓
                              LOCAL OFFLINE TTS
                        (16-bit PCM WAV Synthesis)
                                      ↓
                           BROWSER & SPEAKER PLAYBACK
                     (HTML5 Audio / Web Audio API Out)
                                      ↺
                       DYNAMIC RESOURCE CONTROLLER
            (psutil closed-loop host monitoring: CPU, RAM, Latency)
            → Dynamically adapts AI Threads (8 → 4 → 2) and degradation mode
```

---

## 5. Technology Stack

| Layer | Component | Implementation / Model | Execution Target | Footprint |
|---|---|---|---|---|
| **Audio Capture** | Microphone I/O | `sounddevice` (WASAPI, 16 kHz Mono) + HTML5 MediaRecorder | On-Device Direct I/O | 0 MB |
| **VAD** | Voice Activity Detection | `silero-vad` (PyTorch CPU evaluation mode) | CPU Only | ~11 MB |
| **Local ASR** | Multilingual Speech-to-Text | `whisper.cpp` (GGML tiny binary) / `faster-whisper` (int8) | CPU AVX2 | ~77 MB |
| **Semantic Router** | Dense Vector Embeddings | `all-MiniLM-L6-v2` (`sentence-transformers` CPU) | CPU Only | ~80 MB |
| **Local LLM** | Quantized Reasoning Engine | `Qwen2.5-0.5B-Instruct-Q4_K_M` (`llama-cpp-python`) | CPU AVX2 | ~490 MB |
| **Local TTS** | Speech Synthesis | SAPI5 / `pyttsx3` offline WAV generator | Native Windows CPU | Built-in (0 MB) |
| **Telemetry** | Resource Controller | `psutil` + `ResourceController` closed-loop thread manager | CPU Background Loop | <1 ms overhead |
| **Backend** | API & Audio Server | `FastAPI` + `Uvicorn` asynchronous REST service | Localhost (`127.0.0.1`) | Minimal |
| **Frontend** | Edge Dashboard | Vanilla HTML5, CSS3, JavaScript (Web Audio API) | Any modern browser | Zero build step |

**Total Process Working Set (RAM)**: ~1.2 – 1.6 GB RAM.

---

## 6. Installation & Prerequisites

### Prerequisites
- **OS**: Windows 10/11 (or Linux/macOS with standard audio drivers)
- **Python**: 3.10 – 3.12 (64-bit)
- **Microphone & Speaker**: Default system audio devices enabled

### Step 1: Clone Repository
```powershell
git clone <repository-url>
cd Hacknex
```

### Step 2: Set Up Virtual Environment & Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 3: Verify Local Models
The system automatically uses pre-cached local models stored in `models/`:
- `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (~490 MB)
- `models/whisper/` (CTranslate2 int8 model)
- `models/whispercpp/` (GGML tiny binary)
- `models/embeddings/` (`all-MiniLM-L6-v2`)

*(If downloading fresh models on a clean machine, run `python run.py --download-models` or allow the initial automated download.)*

---

## 7. How to Run & Test

### Start the Application & Web Dashboard
```powershell
.\.venv\Scripts\python.exe run.py --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

### How to Test Voice Input & Real Voice Output
1. Click **"Unlock Speaker / Test Audio"** on the top notification bar to satisfy browser autoplay security policies.
2. Click **"Start Voice"** and speak into your microphone (e.g., *"Hello"* or *"What are your working hours?"*).
3. Click **"Stop"** (or let Silero VAD detect end-of-speech automatically).
4. **Listen**: The synthesized 16-bit PCM audio will play out of your default computer speaker automatically.
5. You can replay any synthesized audio at any time by clicking the **"Replay Audio"** button on the conversation card or header.

### How to Test Dynamic CPU Adaptation (Live Stress Test)
1. On the dashboard, locate the **"Resource Adaptation"** panel.
2. Click **"Resource Stress Test"**.
3. A safe, bounded 7-second background thread generates controlled CPU pressure.
4. **Observe**:
   - System CPU increases to ~60–80%.
   - Operating mode transitions from `Performance` $\to$ `Resource-Safe` or `Balanced`.
   - **AI Threads dynamically scale down from 8 $\to$ 4 or 2**.
   - Dashboard status updates in real time.
5. After 7 seconds, the test safely terminates, and the system automatically normalizes.

### How to Test Multilingual ASR
1. Select your desired input language from the dropdown (or leave on **Auto Detect**).
2. Say a phrase in Spanish (*"Hola como estas"*), Hindi, or French.
3. The dashboard displays the detected language and transcription with native confidence scores.

### How to Run Automated Verification Test Suite
Execute all 37 comprehensive unit, integration, and ablation tests:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
*(All 37 tests execute locally on CPU and complete in ~18 seconds.)*

---

## 8. Real Benchmark Measurements

*Measurements gathered on 13th Gen Intel Core i7-13620H (16 logical cores, Windows 11, CPU-only execution, zero GPU)*:

### Benchmark 1: Tiered Semantic Cache Hit vs LLM Invocation
| Query Type | Routing Decision | Similarity Score | LLM Compute | TTFA (Time to First Audio) | Total Latency |
|---|---|---|---|---|---|
| **Exact Match** (*"Hello"*) | `CACHE HIT (EXACT)` | `1.0000` | **0 tokens (Bypassed)** | **118.7 ms** | **120.4 ms** |
| **Semantic Paraphrase** (*"What are your working hours?"*) | `CACHE HIT (SEMANTIC)` | `0.8842` | **0 tokens (Bypassed)** | **142.3 ms** | **145.1 ms** |
| **Cache Miss** (*"Why is Martian sky reddish?"*) | `CACHE MISS (LLM)` | `0.1689` | **40 tokens generated** | **1064.6 ms** | **1402.1 ms** |

*Benefit: Cache hits save 100% of LLM compute and deliver voice output >8.5x faster.*

### Benchmark 2: Dynamic Thread Scaling & Graceful Degradation
| Resource Mode | System Load Trigger | AI Threads Assigned | Max Tokens Capped | Measured LLM Latency | Machine Responsiveness |
|---|---|---|---|---|---|
| **Performance** | < 50% CPU | **8 Threads** | 80 | ~1250 ms | Peak AI performance |
| **Balanced** | 50–70% CPU | **4 Threads** | 40 | ~820 ms | Balanced load |
| **Survival** | > 85% CPU | **2 Threads** | 18 | ~260 ms | High responsiveness maintained |

---

## 9. Verification of Zero Cloud Dependencies

To verify that the system runs 100% locally and does not leak data:
1. **Source Code Audit**: No imports of `openai`, `anthropic`, `google.generativeai`, `azure`, `boto3`, or cloud speech APIs exist in runtime code.
2. **Network Disconnect Test**: Disconnect your machine from Wi-Fi / Ethernet completely. Start `python run.py --port 8000` and perform voice conversations; the system operates without degradation.
3. **Telemetry Status**: The `/api/status` endpoint and frontend top banner explicitly verify:
   ```json
   {
     "cloud_ai": "OFF",
     "local_inference": "ON",
     "execution_mode": "CPU-only"
   }
   ```

---

## 10. Known Limitations

1. **Hardware Thermal Sensors**: On Windows platforms without specialized OEM kernel drivers, `psutil.sensors_temperatures()` returns `None`. Rather than fabricating fake thermal numbers, the system displays `"Not measured (No ACPI thermal driver exposed)"` and adapts via CPU utilization and process pressure.
2. **First-Load Cold Start**: On the very first invocation after boot, initializing the quantized GGUF weights into CPU RAM takes ~1.5–2.0 seconds. Subsequent queries run instantly from memory.
3. **Browser Audio Autoplay**: Modern web browsers block unprompted audio autoplay until a user gesture occurs. AdaptiveEdge handles this cleanly with an **"Unlock Speaker / Test Audio"** banner and interactive microphone trigger.
