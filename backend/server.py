"""
FastAPI Backend Server for AdaptiveEdge On-Device Conversational AI.
Serves telemetry API, pipeline control, audio stream endpoints, mode switching,
benchmark / ablation engine, stress testing, and modern technical dashboard.
"""

import os
import io
import time
import tempfile
import numpy as np
from scipy.io import wavfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, PlainTextResponse
from pydantic import BaseModel
from typing import Optional

from backend.pipeline import VoicePipeline
from monitoring.resource_monitor import monitor
from cache.semantic_cache import semantic_cache
from monitoring.resource_controller import resource_controller
from monitoring.benchmark_engine import benchmark_engine
from monitoring.stress_tester import stress_tester

app = FastAPI(title="AdaptiveEdge Voice AI API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline: Optional[VoicePipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    print("[Server] Starting AdaptiveEdge On-Device AI backend...")
    if pipeline is None:
        pipeline = VoicePipeline(enable_speaker=True, enable_streaming_tts=True)
    print("[Server] Ready to receive voice, multilingual, and benchmark requests.")


@app.on_event("shutdown")
def shutdown_event():
    global pipeline
    if pipeline and pipeline.is_listening:
        pipeline.stop_listening()


# Request / Response Schemas
class TextQueryRequest(BaseModel):
    text: str


class SpeakerToggleRequest(BaseModel):
    enabled: bool


class StreamingToggleRequest(BaseModel):
    enabled: bool


class ModeChangeRequest(BaseModel):
    mode: str  # "Performance" | "Balanced" | "Resource-Safe" | "Survival" | "Constrained" | "Auto"


class ASRBackendRequest(BaseModel):
    backend: str  # "whisper.cpp" | "faster-whisper"


@app.get("/api/status")
def get_status():
    """Returns real-time system metrics, hardware specs, controller state, and pipeline state."""
    metrics = monitor.get_system_metrics()
    cache_stats = monitor.get_cache_stats()
    pipe_state = pipeline.get_state() if pipeline else {}
    ctrl_state = resource_controller.get_controller_state()

    # Sync metrics operating mode with controller
    metrics["operating_mode"] = ctrl_state["current_mode"]

    return {
        "status": "ONLINE",
        "system": metrics,
        "cache": cache_stats,
        "controller": ctrl_state,
        "pipeline": pipe_state
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns recent latency turns and cache performance statistics."""
    return {
        "cache": monitor.get_cache_stats(),
        "recent_turns": monitor.get_recent_turns(limit=10)
    }


@app.post("/api/settings/mode")
def set_mode(req: ModeChangeRequest):
    """Change operating mode between Performance, Balanced, Resource-Safe, Survival, Constrained, or Auto."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if req.mode.lower() == "auto":
        pipeline.resource_controller.set_auto_mode()
    else:
        pipeline.resource_controller.set_mode(req.mode, manual=True)

    # Immediately apply thread allocation
    pipeline._apply_active_threads()

    return {
        "status": "updated",
        "controller": pipeline.resource_controller.get_controller_state()
    }


@app.post("/api/settings/asr")
def set_asr_backend(req: ASRBackendRequest):
    """Switch active local ASR engine between whisper.cpp and faster-whisper."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.asr.backend_choice = req.backend
    pipeline.asr._init_backend()
    return {"status": "updated", "active_backend": pipeline.asr.active_backend}


@app.post("/api/settings/streaming")
def toggle_streaming(req: StreamingToggleRequest):
    """Toggle sentence-level streaming TTS on or off."""
    if pipeline:
        pipeline.enable_streaming_tts = req.enabled
    return {"streaming_tts_enabled": pipeline.enable_streaming_tts if pipeline else req.enabled}


@app.post("/api/settings/speaker")
def toggle_speaker(req: SpeakerToggleRequest):
    """Enable or disable physical speaker playback."""
    if pipeline:
        pipeline.enable_speaker = req.enabled
        if not req.enabled and hasattr(pipeline.tts, "stop_audio"):
            pipeline.tts.stop_audio()
    return {"speaker_enabled": req.enabled}


@app.post("/api/voice/start")
def start_voice():
    """Start local microphone listening loop."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.start_listening()
    return {"status": "started", "is_listening": True}


@app.post("/api/voice/stop")
def stop_voice():
    """Stop local microphone listening loop."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.stop_listening()
    return {"status": "stopped", "is_listening": False}


@app.post("/api/pipeline/process_text")
def process_text(req: TextQueryRequest):
    """
    Process text query through Router -> (Cache | LLM) -> TTS.
    Returns full T0..T5 latency breakdown, semantic similarity info, and audio payloads.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    turn_data = pipeline.process_text_query(req.text)
    return turn_data


@app.post("/api/pipeline/process_audio")
async def process_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    translate: bool = Form(False)
):
    """
    Process uploaded speech audio (WAV format, 16kHz) through full multilingual stack:
    ASR -> Language Identification -> Router -> (Cache | LLM) -> Streaming TTS -> Speaker.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    contents = await file.read()
    try:
        sr, audio_data = wavfile.read(io.BytesIO(contents))
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        if audio_float.ndim > 1:
            audio_float = audio_float[:, 0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV audio data: {e}")

    turn_data = pipeline.process_speech_audio(
        audio_float,
        speech_end_ts=time.time(),
        language=language,
        translate=translate
    )
    return turn_data


@app.post("/api/stress_test/start")
def start_stress():
    """Start bounded safe CPU stress test for adaptation demonstration."""
    return stress_tester.start()


@app.post("/api/stress_test/stop")
def stop_stress():
    """Stop CPU stress test."""
    return stress_tester.stop()


@app.get("/api/stress_test/status")
def get_stress_status():
    """Get status of stress test."""
    return stress_tester.get_status()


@app.post("/api/benchmark/run")
def run_benchmark():
    """Execute live comprehensive empirical benchmark comparing Baseline vs AdaptiveEdge."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    results = benchmark_engine.run_comprehensive_benchmark(pipeline)
    return results


@app.get("/api/benchmark/results")
def get_benchmark_results():
    """Get the latest benchmark results."""
    if not benchmark_engine.last_benchmark_results:
        if pipeline:
            return benchmark_engine.run_comprehensive_benchmark(pipeline)
        raise HTTPException(status_code=404, detail="Benchmark not run yet.")
    return benchmark_engine.last_benchmark_results


@app.get("/api/benchmark/export")
def export_benchmark(format: str = Query("json", regex="^(json|csv)$")):
    """Export benchmark comparative results as JSON or CSV."""
    if not benchmark_engine.last_benchmark_results and pipeline:
        benchmark_engine.run_comprehensive_benchmark(pipeline)

    if format == "csv":
        csv_data = benchmark_engine.export_csv()
        return Response(content=csv_data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=adaptiveedge_benchmark.csv"})
    else:
        return JSONResponse(content=benchmark_engine.last_benchmark_results)


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    """Serve synthesized TTS audio files."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    file_path = os.path.join(pipeline.tts.temp_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")


# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
