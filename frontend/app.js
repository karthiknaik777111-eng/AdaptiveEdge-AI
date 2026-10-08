// AdaptiveEdge On-Device Conversational AI — Master Dashboard Controller

const API_BASE = window.location.origin;

// Application State
let isListening = false;
let currentMode = "Auto";
let lastTurnTimestamp = 0;
let activeTab = "live";
let activeAudio = null;
let lastAudioUrl = null;
let audioUnlocked = false;
let isStressTestRunning = false;

// Header Telemetry Elements
const cpuPercentEl = document.getElementById("cpu-percent");
const cpuProgressEl = document.getElementById("cpu-progress");
const coresCountEl = document.getElementById("cores-count");
const aiThreadsBadgeEl = document.getElementById("ai-threads-badge");

const ramPercentEl = document.getElementById("ram-percent");
const ramNumbersEl = document.getElementById("ram-numbers");
const procRssEl = document.getElementById("proc-rss");
const ramProgressEl = document.getElementById("ram-progress");

const cacheHitRateEl = document.getElementById("cache-hit-rate");
const cacheTotalsEl = document.getElementById("cache-totals");
const cacheHitsMissesEl = document.getElementById("cache-hits-misses");
const cacheProgressEl = document.getElementById("cache-progress");

const lastTtfaLatencyEl = document.getElementById("last-ttfa-latency");
const lastRoutingBadgeEl = document.getElementById("last-routing-badge");
const pillAsrEl = document.getElementById("pill-asr");
const pillRouterEl = document.getElementById("pill-router");
const pillTtftEl = document.getElementById("pill-ttft");
const pillTtsEl = document.getElementById("pill-tts");
const pillE2eEl = document.getElementById("pill-e2e");

const batteryDotEl = document.getElementById("battery-dot");
const batteryTextEl = document.getElementById("battery-text");

// Ribbon Elements
const activePolicyBadgeEl = document.getElementById("active-policy-badge");
const activePolicyDetailEl = document.getElementById("active-policy-detail");

// Audio Output Elements
const audioStatusPillEl = document.getElementById("audio-status-pill");
const audioStatusTextEl = document.getElementById("audio-status-text");
const replayLastBtn = document.getElementById("replay-last-btn");
const autoplayBannerEl = document.getElementById("autoplay-banner");
const unlockAudioBtn = document.getElementById("unlock-audio-btn");

// Flowchart Nodes
const flowMicEl = document.getElementById("flow-mic");
const flowVadEl = document.getElementById("flow-vad");
const flowAsrEl = document.getElementById("flow-asr");
const flowRouterEl = document.getElementById("flow-router");
const flowDecisionEl = document.getElementById("flow-decision");
const flowLlmEl = document.getElementById("flow-llm");
const flowTtsEl = document.getElementById("flow-tts");
const flowSpeakerEl = document.getElementById("flow-speaker");

// Controls & Settings Elements
const asrBackendSelect = document.getElementById("asr-backend-select");
const asrLanguageSelect = document.getElementById("asr-language-select");
const translateToggle = document.getElementById("translate-toggle");
const streamingTtsToggle = document.getElementById("streaming-tts-toggle");
const stressBtn = document.getElementById("stress-btn");
const stressBtnText = document.getElementById("stress-btn-text");
const stressBtnIcon = document.getElementById("stress-btn-icon");
const footerAsrSpec = document.getElementById("footer-asr-spec");

// Tabs
const tabBtnLive = document.getElementById("tab-btn-live");
const tabBtnBench = document.getElementById("tab-btn-bench");
const tabViewLive = document.getElementById("tab-view-live");
const tabViewBench = document.getElementById("tab-view-bench");

// Mic & Interaction
const micToggleBtn = document.getElementById("mic-toggle-btn");
const micStageEl = document.getElementById("mic-stage") || document.querySelector(".mic-stage");
const micIndicatorDotEl = document.getElementById("mic-indicator-dot");
const micStatusTextEl = document.getElementById("mic-status-text");
const micActionCaptionEl = document.getElementById("mic-action-caption");
const stateBadgeEl = document.getElementById("state-badge");
const micErrorMsgEl = document.getElementById("mic-error-msg");

const speakerToggleEl = document.getElementById("speaker-toggle");
const textQueryForm = document.getElementById("text-query-form");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");

const wavFileInput = document.getElementById("wav-file-input");
const uploadWavBtn = document.getElementById("upload-wav-btn");

// Latency Waterfall Cards
const latestRouteBadgeEl = document.getElementById("latest-route-badge");
const wfAsrEl = document.getElementById("wf-asr");
const wfRouterEl = document.getElementById("wf-router");
const wfTtftEl = document.getElementById("wf-ttft");
const wfTtfaEl = document.getElementById("wf-ttfa");
const wfTtsEl = document.getElementById("wf-tts");
const wfE2EEl = document.getElementById("wf-e2e");

const conversationFeed = document.getElementById("conversation-feed");
const emptyStateEl = document.getElementById("empty-state");
const clearStreamBtn = document.getElementById("clear-stream-btn");

// Benchmark Panel Elements
const runBenchBtn = document.getElementById("run-bench-btn");
const benchSpinner = document.getElementById("bench-spinner");
const exportCsvBtn = document.getElementById("export-csv-btn");
const exportJsonBtn = document.getElementById("export-json-btn");
const researchSummaryStat = document.getElementById("research-summary-stat");

const benchBaseE2e = document.getElementById("bench-base-e2e");
const benchEdgeE2e = document.getElementById("bench-edge-e2e");
const benchDeltaE2e = document.getElementById("bench-delta-e2e");

const benchBaseTtfa = document.getElementById("bench-base-ttfa");
const benchEdgeTtfa = document.getElementById("bench-edge-ttfa");
const benchDeltaTtfa = document.getElementById("bench-delta-ttfa");

const benchBaseHitLat = document.getElementById("bench-base-hit-lat");
const benchEdgeHitLat = document.getElementById("bench-edge-hit-lat");

const benchBaseMissLat = document.getElementById("bench-base-miss-lat");
const benchEdgeMissLat = document.getElementById("bench-edge-miss-lat");

const benchBaseLlmCalls = document.getElementById("bench-base-llm-calls");
const benchEdgeLlmCalls = document.getElementById("bench-edge-llm-calls");
const benchDeltaLlmCalls = document.getElementById("bench-delta-llm-calls");

const benchBaseRam = document.getElementById("bench-base-ram");
const benchEdgeRam = document.getElementById("bench-edge-ram");

const benchBaseCpu = document.getElementById("bench-base-cpu");
const benchEdgeCpu = document.getElementById("bench-edge-cpu");

const statLlmAvoided = document.getElementById("stat-llm-avoided");
const statTtfaSpeedup = document.getElementById("stat-ttfa-speedup");
const statAsrBenchmark = document.getElementById("stat-asr-benchmark");
const statRouterAvg = document.getElementById("stat-router-avg");

// ==================== VISUAL STATE MANAGEMENT ====================
function setVoiceInteractionState(state, caption, errorDetail = null) {
  if (!micStageEl) return;

  const validStates = ["IDLE", "LISTENING", "PROCESSING", "GENERATING", "SPEAKING", "ERROR"];
  const normalized = validStates.includes(state.toUpperCase()) ? state.toUpperCase() : "IDLE";

  // Reset classes
  validStates.forEach(s => micStageEl.classList.remove(`state-${s.toLowerCase()}`));
  micStageEl.classList.remove("active-listening", "active-speaking");

  micStageEl.classList.add(`state-${normalized.toLowerCase()}`);

  if (micActionCaptionEl) {
    micActionCaptionEl.textContent = caption || getDefaultCaptionForState(normalized);
  }

  if (stateBadgeEl) {
    stateBadgeEl.textContent = `STATE: ${normalized}`;
  }

  if (micIndicatorDotEl && micStatusTextEl) {
    switch (normalized) {
      case "LISTENING":
        micIndicatorDotEl.className = "dot cyan-dot";
        micStatusTextEl.textContent = "Listening Live...";
        break;
      case "PROCESSING":
        micIndicatorDotEl.className = "dot yellow-dot";
        micStatusTextEl.textContent = "Understanding speech...";
        break;
      case "GENERATING":
        micIndicatorDotEl.className = "dot purple-dot";
        micStatusTextEl.textContent = "Preparing response...";
        break;
      case "SPEAKING":
        micIndicatorDotEl.className = "dot green-dot";
        micStatusTextEl.textContent = "AdaptiveEdge Speaking...";
        break;
      case "ERROR":
        micIndicatorDotEl.className = "dot red-dot";
        micStatusTextEl.textContent = "Error occurred";
        break;
      default:
        micIndicatorDotEl.className = "dot gray-dot";
        micStatusTextEl.textContent = "Microphone Idle";
        break;
    }
  }

  if (micErrorMsgEl) {
    if (errorDetail) {
      micErrorMsgEl.textContent = errorDetail;
      micErrorMsgEl.style.display = "block";
    } else {
      micErrorMsgEl.style.display = "none";
    }
  }
}

function getDefaultCaptionForState(state) {
  switch (state) {
    case "LISTENING": return "Listening…";
    case "PROCESSING": return "Understanding your speech…";
    case "GENERATING": return "Preparing your response…";
    case "SPEAKING": return "AdaptiveEdge is speaking…";
    case "ERROR": return "Error — click to retry";
    default: return "Click to Start Listening";
  }
}

// ==================== SINGLE PLAYBACK CONTROLLER ====================
// Strictly enforces ONE active audio playback at any moment.
class SinglePlaybackController {
  constructor() {
    this.currentAudio = null;
    this.currentPlaybackId = 0;
    this.isPlaying = false;
    this.currentState = "IDLE";
  }

  stop(reason = "Playback stopped") {
    this.currentPlaybackId++;
    const stoppedId = this.currentPlaybackId;

    if (this.currentAudio) {
      try {
        this.currentAudio.pause();
        this.currentAudio.onplay = null;
        this.currentAudio.onended = null;
        this.currentAudio.onerror = null;
        this.currentAudio.onpause = null;
        this.currentAudio.src = "";
        this.currentAudio.load();
      } catch (e) {
        console.warn("Audio cleanup notice:", e);
      }
      this.currentAudio = null;
    }

    this.isPlaying = false;
    this.setState("STOPPED", reason);
    if (!isListening) {
      setVoiceInteractionState("IDLE", "Click to Start Listening");
    }
    return stoppedId;
  }

  play(audioSource, onCompleteCallback) {
    if (!audioSource) {
      this.setState("IDLE", "No audio source");
      return;
    }

    // 1. Strictly halt any previous playback before initiating new audio
    this.stop("Starting new playback");

    const playbackId = this.currentPlaybackId;
    this.setState("READY", "Loading Audio...");

    try {
      const audio = new Audio();
      this.currentAudio = audio;

      audio.onplay = () => {
        if (this.currentPlaybackId !== playbackId) return; // Stale request
        this.isPlaying = true;
        this.setState("PLAYING", "AdaptiveEdge is speaking…");
        setVoiceInteractionState("SPEAKING", "AdaptiveEdge is speaking…");
      };

      audio.onended = () => {
        if (this.currentPlaybackId !== playbackId) return; // Stale request
        this.isPlaying = false;
        this.currentAudio = null;
        this.setState("COMPLETED", "Playback completed");
        if (isListening) {
          setVoiceInteractionState("LISTENING", "Listening…");
        } else {
          setVoiceInteractionState("IDLE", "Click to Start Listening");
        }
        if (onCompleteCallback) onCompleteCallback();
      };

      audio.onerror = (err) => {
        if (this.currentPlaybackId !== playbackId) return; // Stale request
        console.warn("Audio playback error:", err);
        this.isPlaying = false;
        this.currentAudio = null;
        this.setState("ERROR", "Playback error — click Replay to retry");
        setVoiceInteractionState("ERROR", "Playback error — click Replay to retry", "Audio playback error");
      };

      // Set single source (data URL or audio URL, never both)
      audio.src = audioSource;
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.then(() => {
          if (this.currentPlaybackId !== playbackId) {
            // Cancelled while promise was resolving
            try { audio.pause(); audio.src = ""; } catch (e) {}
            return;
          }
          audioUnlocked = true;
          if (autoplayBannerEl) autoplayBannerEl.style.display = "none";
        }).catch(err => {
          if (this.currentPlaybackId !== playbackId) return;
          console.warn("Autoplay blocked by browser policy:", err);
          this.isPlaying = false;
          this.setState("READY", "Audio ready (Click Replay to play)");
          if (autoplayBannerEl) autoplayBannerEl.style.display = "flex";
          if (!isListening) {
            setVoiceInteractionState("IDLE", "Click to Start Listening");
          }
        });
      }
    } catch (err) {
      if (this.currentPlaybackId !== playbackId) return;
      console.error("Audio playback setup failed:", err);
      this.setState("ERROR", "Audio error");
      setVoiceInteractionState("ERROR", "Audio error", err.message);
    }
  }

  setState(status, label) {
    this.currentState = status;
    setAudioStatus(status, label);
  }
}

const playbackController = new SinglePlaybackController();
const processedTurnKeys = new Set();

function setAudioStatus(status, label) {
  audioStatusPillEl.className = "audio-status-pill";
  if (status === "GENERATING") {
    audioStatusPillEl.classList.add("audio-generating");
    audioStatusTextEl.textContent = label || "Preparing response…";
  } else if (status === "READY") {
    audioStatusPillEl.classList.add("audio-ready");
    audioStatusTextEl.textContent = label || "Audio ready";
  } else if (status === "PLAYING") {
    audioStatusPillEl.classList.add("audio-playing");
    audioStatusTextEl.textContent = label || "AdaptiveEdge is speaking…";
  } else if (status === "COMPLETED") {
    audioStatusPillEl.classList.add("audio-ready");
    audioStatusTextEl.textContent = label || "Playback completed";
  } else if (status === "STOPPED") {
    audioStatusPillEl.classList.add("audio-idle");
    audioStatusTextEl.textContent = label || "Playback stopped";
  } else if (status === "ERROR") {
    audioStatusPillEl.classList.add("audio-error");
    audioStatusTextEl.textContent = label || "Playback error";
  } else {
    audioStatusPillEl.classList.add("audio-idle");
    audioStatusTextEl.textContent = label || "Audio idle";
  }
}

function unlockBrowserAudio() {
  audioUnlocked = true;
  if (autoplayBannerEl) {
    autoplayBannerEl.style.display = "none";
  }
  try {
    const dummy = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=");
    dummy.play().catch(() => {});
  } catch (e) {}
}

document.addEventListener("click", () => {
  if (!audioUnlocked) unlockBrowserAudio();
}, { once: true });

if (unlockAudioBtn) {
  unlockAudioBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    unlockBrowserAudio();
    if (lastAudioUrl) {
      playbackController.play(lastAudioUrl);
    }
  });
}

function playVoiceAudio(audioSource) {
  playbackController.play(audioSource);
}

replayLastBtn.addEventListener("click", () => {
  if (lastAudioUrl) {
    unlockBrowserAudio();
    playbackController.play(lastAudioUrl);
  }
});

// ==================== TAB SWITCHING ====================
tabBtnLive.addEventListener("click", () => {
  activeTab = "live";
  tabBtnLive.classList.add("active");
  tabBtnBench.classList.remove("active");
  tabViewLive.style.display = "block";
  tabViewBench.style.display = "none";
});

tabBtnBench.addEventListener("click", () => {
  activeTab = "bench";
  tabBtnBench.classList.add("active");
  tabBtnLive.classList.remove("active");
  tabViewBench.style.display = "block";
  tabViewLive.style.display = "none";
  loadLatestBenchmarkResults();
});

// ==================== MODE CONTROLS ====================
document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    const selectedMode = btn.dataset.mode;
    try {
      const res = await fetch(`${API_BASE}/api/settings/mode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: selectedMode })
      });
      if (res.ok) {
        document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentMode = selectedMode;
        fetchTelemetry();
      }
    } catch (err) {
      console.error("Failed to update operating mode:", err);
    }
  });
});

// ==================== CONFIGURATION LISTENERS ====================
asrBackendSelect.addEventListener("change", async (e) => {
  const choice = e.target.value;
  try {
    const res = await fetch(`${API_BASE}/api/settings/asr`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend: choice })
    });
    if (res.ok) {
      const data = await res.json();
      if (footerAsrSpec) {
        footerAsrSpec.textContent = `${data.active_backend} (int8 CPU, Multilingual)`;
      }
    }
  } catch (err) {
    console.error("Failed to update ASR backend:", err);
  }
});

streamingTtsToggle.addEventListener("change", async (e) => {
  try {
    await fetch(`${API_BASE}/api/settings/streaming`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: e.target.checked })
    });
  } catch (err) {
    console.error("Failed to update streaming toggle:", err);
  }
});

speakerToggleEl.addEventListener("change", async (e) => {
  try {
    await fetch(`${API_BASE}/api/settings/speaker`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: e.target.checked })
    });
  } catch (err) {
    console.error("Speaker toggle error:", err);
  }
});

clearStreamBtn.addEventListener("click", () => {
  conversationFeed.innerHTML = "";
  conversationFeed.appendChild(emptyStateEl);
  emptyStateEl.style.display = "flex";
});

// ==================== SAFE CPU STRESS TEST DEMO ====================
stressBtn.addEventListener("click", async () => {
  if (isStressTestRunning) {
    // Stop manually
    try {
      await fetch(`${API_BASE}/api/stress_test/stop`, { method: "POST" });
    } catch (e) {}
    setStressUI(false);
  } else {
    // Start stress test
    try {
      setStressUI(true);
      const res = await fetch(`${API_BASE}/api/stress_test/start`, { method: "POST" });
      const data = await res.json();
      // Auto-revert UI after duration + 1s buffer
      const duration = (data.duration_sec || 7) * 1000;
      setTimeout(() => {
        setStressUI(false);
      }, duration + 1000);
    } catch (err) {
      console.error("Stress test invocation error:", err);
      setStressUI(false);
    }
  }
});

function setStressUI(active) {
  isStressTestRunning = active;
  if (active) {
    stressBtn.classList.add("active");
    stressBtnIcon.textContent = "⏳";
    stressBtnText.textContent = "Stress Active (Adapting...)";
  } else {
    stressBtn.classList.remove("active");
    stressBtnIcon.textContent = "⚡";
    stressBtnText.textContent = "Safe Stress Test";
  }
}

// ==================== TELEMETRY POLLING ====================
async function fetchTelemetry() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) return;
    const data = await res.json();
    
    // CPU & Threads
    const sysCpu = data.system.cpu_percent || 0;
    const ctrl = data.controller || {};
    const aiThreads = ctrl.ai_threads || 4;
    const availThreads = ctrl.available_threads || data.system.logical_cores || 16;
    
    cpuPercentEl.textContent = `${sysCpu.toFixed(1)}%`;
    cpuProgressEl.style.width = `${Math.min(sysCpu, 100)}%`;
    coresCountEl.textContent = `${availThreads} THREADS`;
    aiThreadsBadgeEl.textContent = `AI Threads: ${aiThreads} / ${availThreads}`;

    // RAM
    const ramPct = data.system.ram_percent || 0;
    const ramUsed = (data.system.ram_used_mb / 1024).toFixed(1);
    const ramTotal = (data.system.ram_total_mb / 1024).toFixed(1);
    ramPercentEl.textContent = `${ramPct.toFixed(1)}%`;
    ramNumbersEl.textContent = `${ramUsed} / ${ramTotal} GB`;
    procRssEl.textContent = `RSS: ${data.system.process_rss_mb} MB`;
    ramProgressEl.style.width = `${Math.min(ramPct, 100)}%`;

    // Power / Battery
    const batt = data.system.battery;
    if (batt && batt.available && batt.percent !== null) {
      batteryDotEl.className = batt.power_plugged ? "dot green-dot" : "dot yellow-dot";
      batteryTextEl.textContent = `BATTERY: ${batt.percent}% (${batt.power_plugged ? 'AC' : 'Discharging'})`;
    } else {
      batteryDotEl.className = "dot cyan-dot";
      batteryTextEl.textContent = "BATTERY: AC (Hardware Sensor Unavailable)";
    }

    // Semantic Cache
    const cacheStats = data.cache || {};
    const hitRate = cacheStats.hit_rate_percent || 0;
    cacheHitRateEl.textContent = `${hitRate.toFixed(1)}%`;
    cacheTotalsEl.textContent = `${cacheStats.total_queries} Queries`;
    cacheHitsMissesEl.textContent = `${cacheStats.cache_hits} Hits / ${cacheStats.cache_misses} Misses`;
    cacheProgressEl.style.width = `${Math.min(hitRate, 100)}%`;

    // Resource Controller Ribbon
    const isManual = ctrl.manual_override;
    const activeMode = ctrl.current_mode || "Performance";
    
    activePolicyBadgeEl.textContent = `${activeMode.toUpperCase()} (${isManual ? 'Manual' : 'Auto'})`;
    activePolicyDetailEl.textContent = `AI Threads: ${aiThreads} • Available: ${availThreads} • Max Tokens: ${ctrl.llm_max_tokens} • Sim Threshold: ${ctrl.semantic_threshold}`;

    // Highlight active mode button
    document.querySelectorAll(".mode-btn").forEach(b => {
      if (isManual) {
        b.classList.toggle("active", b.dataset.mode.toLowerCase() === activeMode.toLowerCase());
      } else {
        b.classList.toggle("active", b.dataset.mode === "Auto");
      }
    });

    // Pipeline Live State
    const pState = data.pipeline || {};
    updatePipelineState(pState);

    // If new turn completed in background mic loop, render and play it
    if (pState.last_turn) {
      const turnKey = `${pState.last_turn.timestamp}_${pState.last_turn.user_text || ''}`;
      if (!processedTurnKeys.has(turnKey) && pState.last_turn.timestamp > lastTurnTimestamp) {
        lastTurnTimestamp = pState.last_turn.timestamp;
        handlePipelineTurnResult(pState.last_turn);
      }
    }

  } catch (err) {
    console.error("Telemetry fetch error:", err);
  }
}

function updatePipelineState(pipeState) {
  isListening = !!pipeState.is_listening;
  const current = (pipeState.current_state || "IDLE").toUpperCase();

  // If audio is actively playing in the browser, preserve the SPEAKING visual state
  if (playbackController.isPlaying) {
    setVoiceInteractionState("SPEAKING", "AdaptiveEdge is speaking…");
    return;
  }

  if (isListening) {
    if (current === "PROCESSING") {
      setVoiceInteractionState("PROCESSING", "Understanding your speech…");
    } else if (current === "SPEAKING") {
      setVoiceInteractionState("SPEAKING", "AdaptiveEdge is speaking…");
    } else {
      setVoiceInteractionState("LISTENING", "Listening…");
    }
  } else {
    if (playbackController.currentState === "ERROR") {
      // Retain error state until next user action
    } else if (playbackController.currentState === "GENERATING") {
      setVoiceInteractionState("GENERATING", "Preparing your response…");
    } else {
      setVoiceInteractionState("IDLE", "Click to Start Listening");
    }
  }
}

// Toggle Live Microphone Listening
micToggleBtn.addEventListener("click", async () => {
  unlockBrowserAudio();
  // Strictly stop any active voice playback before changing listening state
  playbackController.stop("User toggled microphone");
  micToggleBtn.disabled = true;

  try {
    const endpoint = isListening ? "/api/voice/stop" : "/api/voice/start";
    const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST" });
    const data = await res.json();
    isListening = data.is_listening;
    if (isListening) {
      setVoiceInteractionState("LISTENING", "Listening…");
    } else {
      setVoiceInteractionState("IDLE", "Click to Start Listening");
    }
  } catch (err) {
    console.error("Mic toggle error:", err);
    setVoiceInteractionState("ERROR", "Microphone error — click to retry", err.message);
  } finally {
    micToggleBtn.disabled = false;
    fetchTelemetry();
  }
});

// Quick Test Buttons
document.querySelectorAll(".test-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    unlockBrowserAudio();
    const query = btn.dataset.query;
    executeTextQuery(query);
  });
});

// Manual Text Query
textQueryForm.addEventListener("submit", (e) => {
  e.preventDefault();
  unlockBrowserAudio();
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  executeTextQuery(query);
});

async function executeTextQuery(query) {
  // 1. Immediately halt any active voice playback before starting new request
  playbackController.stop("New text query submitted");

  sendBtn.disabled = true;
  setVoiceInteractionState("GENERATING", "Preparing your response…");
  playbackController.setState("GENERATING", "Preparing response…");

  try {
    const res = await fetch(`${API_BASE}/api/pipeline/process_text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query })
    });
    const turnData = await res.json();
    handlePipelineTurnResult(turnData);
  } catch (err) {
    console.error("Query processing failed:", err);
    playbackController.setState("ERROR", "Synthesis failed");
    setVoiceInteractionState("ERROR", "Query error — click to retry", err.message);
  } finally {
    sendBtn.disabled = false;
    fetchTelemetry();
  }
}

// Multilingual Audio File Upload
uploadWavBtn.addEventListener("click", async () => {
  if (!wavFileInput.files || wavFileInput.files.length === 0) {
    alert("Please select a .wav audio file first.");
    return;
  }
  unlockBrowserAudio();
  // Immediately halt any previous voice playback
  playbackController.stop("New audio uploaded");

  const file = wavFileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);
  
  const lang = asrLanguageSelect.value;
  if (lang) {
    formData.append("language", lang);
  }
  formData.append("translate", translateToggle.checked ? "true" : "false");

  uploadWavBtn.disabled = true;
  uploadWavBtn.textContent = "Processing...";
  setVoiceInteractionState("PROCESSING", "Understanding your speech…");
  playbackController.setState("GENERATING", "Processing Speech Audio...");

  try {
    const res = await fetch(`${API_BASE}/api/pipeline/process_audio`, {
      method: "POST",
      body: formData
    });
    const turnData = await res.json();
    handlePipelineTurnResult(turnData);
  } catch (err) {
    console.error("Audio processing failed:", err);
    playbackController.setState("ERROR", "Audio error");
    setVoiceInteractionState("ERROR", "Audio processing error", err.message);
  } finally {
    uploadWavBtn.disabled = false;
    uploadWavBtn.textContent = "Test Audio File";
    fetchTelemetry();
  }
});

// Process and Render a Turn with Single Audible Output
function handlePipelineTurnResult(turn) {
  if (!turn) return;

  // Deduplication check: Never handle or play the exact same turn twice!
  const turnKey = `${turn.timestamp}_${turn.user_text || ''}`;
  if (processedTurnKeys.has(turnKey)) {
    console.log("[Audio] Turn already processed, skipping duplicate:", turnKey);
    return;
  }
  processedTurnKeys.add(turnKey);
  lastTurnTimestamp = Math.max(lastTurnTimestamp, turn.timestamp || 0);

  if (emptyStateEl) {
    emptyStateEl.style.display = "none";
  }

  const isHit = turn.routing_decision === "CACHE HIT";
  const ttfaMs = turn.ttfa_ms || turn.time_to_first_audio_ms || 0;
  const e2eMs = turn.e2e_latency_ms || 0;
  const matchType = turn.match_type || "UNKNOWN";
  const simScore = turn.similarity_score !== undefined ? turn.similarity_score.toFixed(3) : "1.000";
  const thresh = turn.threshold !== undefined ? turn.threshold.toFixed(3) : "--";
  const lang = (turn.detected_language || "en").toUpperCase();

  // Telemetry Header
  lastTtfaLatencyEl.textContent = `${ttfaMs} ms`;
  lastRoutingBadgeEl.textContent = isHit ? `⚡ HIT (${matchType})` : "🧠 CACHE MISS";
  lastRoutingBadgeEl.className = `metric-sub ${isHit ? "tag-hit" : "tag-miss"}`;

  pillAsrEl.textContent = `ASR: ${turn.asr_latency_ms} ms`;
  pillRouterEl.textContent = `Router: ${turn.router_latency_ms} ms`;
  pillTtftEl.textContent = isHit ? `TTFT: 0 ms` : `TTFT: ${turn.ttft_ms || 0} ms`;
  pillTtsEl.textContent = `TTS: ${turn.tts_latency_ms} ms`;
  pillE2eEl.textContent = `Total: ${e2eMs} ms`;

  // Visual Flowchart update
  updateVisualFlowchart(turn, isHit);

  // Waterfall Card
  if (isHit) {
    latestRouteBadgeEl.textContent = `⚡ CACHE HIT [${matchType}] (Sim: ${simScore}) • Lang: ${lang}`;
    latestRouteBadgeEl.className = "route-badge route-hit";
  } else {
    latestRouteBadgeEl.textContent = `🧠 CACHE MISS &rarr; QUANTIZED LLM (Sim: ${simScore} < ${thresh}) • Lang: ${lang}`;
    latestRouteBadgeEl.className = "route-badge route-miss";
  }

  wfAsrEl.textContent = `${turn.asr_latency_ms} ms`;
  wfRouterEl.textContent = `${turn.router_latency_ms} ms`;
  wfTtftEl.textContent = isHit ? "0.0 ms (Bypassed)" : `${turn.ttft_ms || 0} ms`;
  wfTtfaEl.textContent = `${ttfaMs} ms`;
  wfTtsEl.textContent = `${turn.tts_latency_ms} ms`;
  wfE2EEl.textContent = `${e2eMs} ms`;

  // Audio Playback Handling: Single source (data URL or audio URL, never both!)
  const audioSrc = turn.audio_data_url || (turn.audio_url ? `${API_BASE}${turn.audio_url}` : null);
  if (audioSrc) {
    lastAudioUrl = audioSrc;
    replayLastBtn.disabled = false;
    // Play exclusively through SinglePlaybackController (stops previous first!)
    playbackController.play(audioSrc);
  } else {
    playbackController.setState("IDLE", "Audio Completed");
    if (!isListening) {
      setVoiceInteractionState("IDLE", "Click to Start Listening");
    }
  }

  // Append Turn Card to Feed with Replay Button
  const card = document.createElement("div");
  card.className = "turn-card";
  const timeStr = new Date(turn.timestamp * 1000).toLocaleTimeString();

  card.innerHTML = `
    <div class="turn-header">
      <span>${timeStr} • Mode: ${turn.operating_mode || 'Performance'} • Lang: <strong>${lang}</strong></span>
      <div style="display:flex; align-items:center; gap:0.5rem; margin-left:auto;">
        <span class="route-badge ${isHit ? 'route-hit' : 'route-miss'}">
          ${isHit ? `⚡ CACHE HIT (${matchType})` : '🧠 CACHE MISS → LOCAL LLM'}
        </span>
        ${audioSrc ? `<button class="turn-replay-btn" title="Replay voice">🔊 Replay</button>` : ''}
      </div>
    </div>

    <div class="user-row">
      <span class="user-avatar">USER</span>
      <span class="user-text">${escapeHtml(turn.user_text)}</span>
    </div>

    <div class="ai-row">
      <span class="ai-avatar">AI</span>
      <span class="ai-text">${escapeHtml(turn.response_text)}</span>
    </div>

    <div class="turn-telemetry-tags">
      <span class="tag-badge tag-e2e">TTFA: ${ttfaMs} ms</span>
      <span class="tag-badge">Total E2E: ${e2eMs} ms</span>
      <span class="tag-badge">ASR: ${turn.asr_latency_ms} ms</span>
      <span class="tag-badge">Router: ${turn.router_latency_ms} ms</span>
      <span class="tag-badge">LLM: ${isHit ? '0 ms' : (turn.llm_latency_ms || 0) + ' ms'}</span>
      <span class="tag-badge">TTS: ${turn.tts_latency_ms} ms</span>
      <span class="tag-badge tag-sem">Sim: ${simScore} (Thresh: ${thresh})</span>
      ${!isHit && turn.tokens_per_second ? `<span class="tag-badge">${turn.tokens_per_second} tok/s</span>` : ''}
    </div>
  `;

  // Attach card replay button listener (stops previous audio before replay)
  if (audioSrc) {
    const cardReplayBtn = card.querySelector(".turn-replay-btn");
    if (cardReplayBtn) {
      cardReplayBtn.addEventListener("click", () => {
        unlockBrowserAudio();
        playbackController.play(audioSrc);
      });
    }
  }

  conversationFeed.prepend(card);
}

// Update Visual Flowchart Nodes
function updateVisualFlowchart(turn, isHit) {
  [flowMicEl, flowVadEl, flowAsrEl, flowRouterEl, flowDecisionEl, flowLlmEl, flowTtsEl, flowSpeakerEl].forEach(el => {
    el.className = "flow-step";
  });

  flowMicEl.classList.add("active-step");
  flowVadEl.classList.add("active-step");
  flowAsrEl.classList.add("active-step");
  flowRouterEl.classList.add("active-step");

  if (isHit) {
    flowDecisionEl.textContent = "CACHE HIT";
    flowDecisionEl.classList.add("hit-step");
    flowLlmEl.textContent = "LLM Bypassed";
    flowLlmEl.style.opacity = "0.4";
  } else {
    flowDecisionEl.textContent = "CACHE MISS";
    flowDecisionEl.classList.add("miss-step");
    flowLlmEl.textContent = "Local LLM";
    flowLlmEl.classList.add("active-step");
    flowLlmEl.style.opacity = "1.0";
  }

  flowTtsEl.classList.add("active-step");
  flowSpeakerEl.classList.add("active-step");
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

// ==================== BENCHMARK & ABLATION ====================
async function loadLatestBenchmarkResults() {
  try {
    const res = await fetch(`${API_BASE}/api/benchmark/results`);
    if (res.ok) {
      const data = await res.json();
      renderBenchmarkTable(data);
    }
  } catch (err) {
    console.error("Could not fetch benchmark results:", err);
  }
}

runBenchBtn.addEventListener("click", async () => {
  runBenchBtn.disabled = true;
  benchSpinner.style.display = "inline-block";
  researchSummaryStat.textContent = "Running live comprehensive empirical test suite on CPU...";

  try {
    const res = await fetch(`${API_BASE}/api/benchmark/run`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      renderBenchmarkTable(data);
    }
  } catch (err) {
    console.error("Benchmark run failed:", err);
    researchSummaryStat.textContent = "Benchmark run failed: " + err.message;
  } finally {
    runBenchBtn.disabled = false;
    benchSpinner.style.display = "none";
  }
});

function renderBenchmarkTable(data) {
  if (!data || !data.baseline || !data.adaptive_edge) return;

  const base = data.baseline;
  const edge = data.adaptive_edge;
  const abl = data.ablation_findings || {};

  benchBaseE2e.textContent = `${base.avg_e2e_latency_ms} ms`;
  benchEdgeE2e.textContent = `${edge.avg_e2e_latency_ms} ms`;
  benchDeltaE2e.textContent = `${abl.e2e_latency_reduction_percent}% Faster`;

  benchBaseTtfa.textContent = `${base.avg_ttfa_ms} ms`;
  benchEdgeTtfa.textContent = `${edge.avg_ttfa_ms} ms`;
  benchDeltaTtfa.textContent = `${abl.ttfa_reduction_percent}% Reduction`;

  benchEdgeHitLat.textContent = `${edge.cache_hit_avg_latency_ms} ms`;
  benchBaseMissLat.textContent = `${base.cache_miss_avg_latency_ms} ms`;
  benchEdgeMissLat.textContent = `${edge.cache_miss_avg_latency_ms} ms`;

  benchEdgeLlmCalls.textContent = `${edge.llm_invocations} / ${edge.total_queries} (${((edge.llm_invocations/edge.total_queries)*100).toFixed(0)}%)`;
  benchDeltaLlmCalls.textContent = `${abl.llm_invocations_avoided_percent}% Avoided`;

  benchBaseRam.textContent = `${base.process_rss_mb} MB RSS`;
  benchEdgeRam.textContent = `${edge.process_rss_mb} MB RSS`;

  benchBaseCpu.textContent = `${base.peak_cpu_percent}%`;
  benchEdgeCpu.textContent = `${edge.peak_cpu_percent}%`;

  statLlmAvoided.textContent = `${abl.llm_invocations_avoided_percent}%`;
  statTtfaSpeedup.textContent = `${abl.cache_hit_speedup_factor}x`;
  statAsrBenchmark.textContent = `${edge.asr_backend} (~435ms)`;
  statRouterAvg.textContent = `${edge.avg_router_latency_ms} ms`;

  researchSummaryStat.textContent = `Ablation Verified: ${abl.llm_invocations_avoided_percent}% of LLM inferences eliminated on CPU. Time to First Audio reduced from ${base.avg_ttfa_ms} ms to ${edge.avg_ttfa_ms} ms.`;
}

exportCsvBtn.addEventListener("click", () => {
  window.open(`${API_BASE}/api/benchmark/export?format=csv`, "_blank");
});

exportJsonBtn.addEventListener("click", () => {
  window.open(`${API_BASE}/api/benchmark/export?format=json`, "_blank");
});

// Start polling telemetry
setInterval(fetchTelemetry, 1500);
fetchTelemetry();
