"""Chat model factory + token/cost tracking callback."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

from langchain_core.callbacks import BaseCallbackHandler

from trialscope import config


def llm_configured() -> bool:
    """True if a usable LLM endpoint is set: a custom OPENAI_BASE_URL (Ollama, Groq, Gemini...) or a real-looking OpenAI key."""
    if os.getenv("OPENAI_BASE_URL"):
        return True
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key.startswith("sk-") and len(key) > 20 and "..." not in key


@lru_cache(maxsize=4)
def get_chat(model: str | None = None, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    base_url = os.getenv("OPENAI_BASE_URL") or None      # e.g. http://localhost:11434/v1 for a free local Ollama model
    api_key = os.getenv("OPENAI_API_KEY") or ("local" if base_url else None)
    return ChatOpenAI(model=model or config.CHAT_MODEL, temperature=temperature, max_retries=6, timeout=90,
                      base_url=base_url, api_key=api_key)


class UsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                model = (getattr(msg, "response_metadata", None) or {}).get("model_name", config.CHAT_MODEL)
                i, o = um.get("input_tokens", 0), um.get("output_tokens", 0)
                pin, pout = next((v for k, v in config.PRICES.items() if str(model).startswith(k)), (0.0, 0.0))
                with self.lock:
                    self.calls += 1
                    self.input_tokens += i
                    self.output_tokens += o
                    self.cost_usd += (i * pin + o * pout) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 5)}
