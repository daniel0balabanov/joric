"""Configuration dataclasses for joric.

Override defaults via environment variables or a .env file.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class STTConfig:
    model_size: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"   # falls back to int8 at runtime if CUDA unavailable
    language: str | None = None     # None = auto-detect


@dataclass
class LLMConfig:
    # backend is resolved at runtime: "claude" if ANTHROPIC_API_KEY is set, else "ollama"
    backend: str = field(default_factory=lambda: "claude" if os.getenv("ANTHROPIC_API_KEY") else "ollama")
    # Ollama settings
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma3:12b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    # Claude settings
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"))
    # Shared
    system_prompt: str = "You are a helpful assistant. Keep your answers concise — they will be read aloud."
    history_limit: int = 20         # max messages kept in context (rolling)
    temperature: float = 0.7
    max_tokens: int = 512
    request_timeout: float = 60.0


@dataclass
class TTSConfig:
    enabled: bool = True
    api_url: str = field(default_factory=lambda: os.getenv("TTS_API_URL", "http://localhost:9880"))
    ref_audio_path: str = (
        "/home/welxeor/GPT-SoVITS/logs/narrator/5-wav32k/Vo_narr_town_upgrade_stage_01.wav"
    )
    prompt_text: str = "Great heroes can be found even here, in the mud and rain."
    text_lang: str = "en"
    prompt_lang: str = "en"


@dataclass
class AppConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
