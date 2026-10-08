"""
Local Quantized Language Model Service.
Uses llama-cpp-python running Qwen2.5-0.5B-Instruct-Q4_K_M GGUF locally on CPU.
Zero cloud API dependencies. Measures exact token generation latency.
"""

import time
import os
from typing import Dict, Any, Generator, Optional
from llama_cpp import Llama


class LLMService:
    def __init__(
        self,
        model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        n_ctx: int = 1024,
        n_threads: int = 4
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.system_prompt = (
            "You are AdaptiveEdge Voice AI, an on-device assistant running locally on CPU. "
            "Give direct, concise, and natural spoken answers in 1 to 2 sentences. "
            "Never use markdown formatting, bullets, or emojis as your response will be spoken aloud."
        )
        self.llm = None
        self._load_model()

    def _load_model(self):
        """Load quantized GGUF model into memory."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"LLM model file not found at: {self.model_path}")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False
        )

    def set_threads(self, n_threads: int):
        """Dynamically update CPU inference thread allocation."""
        if n_threads <= 0:
            return
        self.n_threads = n_threads
        if self.llm is not None:
            try:
                if hasattr(self.llm, "set_n_threads"):
                    self.llm.set_n_threads(n_threads)
                else:
                    self.llm.n_threads = n_threads
            except Exception as e:
                print(f"[LLM] Notice: thread update: {e}")

    def generate(self, user_prompt: str, max_tokens: int = 80) -> Dict[str, Any]:
        """
        Generate response synchronously, measuring time-to-completion and speed.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        t_start = time.perf_counter()
        
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stream=False
        )
        
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        content = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        tok_per_sec = (completion_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

        return {
            "text": content,
            "latency_ms": round(latency_ms, 2),
            "tokens": completion_tokens,
            "tokens_per_second": round(tok_per_sec, 2),
            "model": "Qwen2.5-0.5B-Instruct-Q4_K_M"
        }

    def generate_stream(self, user_prompt: str, max_tokens: int = 80) -> Generator[Dict[str, Any], None, None]:
        """
        Stream generated tokens progressively, tracking time-to-first-token.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        t_start = time.perf_counter()
        first_token_time = None
        token_count = 0

        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stream=True
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                token_count += 1
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    ttft_ms = (first_token_time - t_start) * 1000.0
                else:
                    ttft_ms = None

                yield {
                    "token": content,
                    "first_token": (token_count == 1),
                    "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                    "elapsed_ms": round((now - t_start) * 1000.0, 2)
                }
