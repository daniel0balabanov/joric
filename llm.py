"""LLM client supporting Ollama (default) and Claude (via ANTHROPIC_API_KEY).

Backend selection:
  - ANTHROPIC_API_KEY set → Claude (anthropic SDK)
  - otherwise            → Ollama via openai-compatible API
"""

import logging
import os
from typing import Iterator

from config import LLMConfig

log = logging.getLogger(__name__)


class LLMClient:
    """Multi-turn conversational LLM client."""

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._history: list[dict] = []
        log.info("LLM backend: %s  model: %s", cfg.backend, self._active_model)

    @property
    def _active_model(self) -> str:
        if self._cfg.backend == "claude":
            return self._cfg.claude_model
        return self._cfg.ollama_model

    def chat(self, user_message: str) -> str:
        """Send a message and return the assistant reply. Maintains history."""
        self._history.append({"role": "user", "content": user_message})
        self._trim_history()

        try:
            reply = self._send()
        except Exception as exc:
            log.error("LLM error: %s", exc)
            reply = f"[LLM error: {exc}]"

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        limit = self._cfg.history_limit
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def _send(self) -> str:
        if self._cfg.backend == "claude":
            return self._send_claude()
        return self._send_ollama()

    def _send_ollama(self) -> str:
        import openai

        client = openai.OpenAI(
            base_url=self._cfg.ollama_base_url,
            api_key="ollama",
        )
        response = client.chat.completions.create(
            model=self._cfg.ollama_model,
            messages=[{"role": "system", "content": self._cfg.system_prompt}, *self._history],
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            timeout=self._cfg.request_timeout,
        )
        return response.choices[0].message.content.strip()

    def _send_claude(self) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=self._cfg.claude_model,
            system=self._cfg.system_prompt,
            messages=self._history,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
        return response.content[0].text.strip()
