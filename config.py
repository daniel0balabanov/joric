"""Configuration dataclasses for joric.

Override defaults via environment variables or a .env file.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from prompts import PERSONAS

load_dotenv()

# ---------------------------------------------------------------------------
# Voice profiles — add trained voices here.
# Each entry: "name": ("/abs/path/to/ref.wav", "transcript of ref clip")
# ---------------------------------------------------------------------------
VOICE_PROFILES: dict[str, tuple[str, str]] = {
    "narrator": (
        "/home/welxeor/GPT-SoVITS/logs/narrator/5-wav32k/Vo_narr_town_upgrade_stage_01.wav",
        "Great heroes can be found even here, in the mud and rain.",
    ),
    # "myvoice": ("/path/to/ref.wav", "transcript of reference clip"),
}


@dataclass
class STTConfig:
    model_size: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3-turbo"))
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None     # None = auto-detect
    beam_size: int = 1              # 1 = greedy (fastest), 5 = more accurate


@dataclass
class LLMConfig:
    # backend priority: gemini > claude > ollama
    backend: str = field(default_factory=lambda: (
        "gemini" if os.getenv("GOOGLE_API_KEY") else
        "claude" if os.getenv("ANTHROPIC_API_KEY") else
        "ollama"
    ))
    # Ollama settings
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma4:e4b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    ollama_num_gpu: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_GPU", "-1")))  # 0=CPU/RAM, -1=auto GPU
    # Claude settings
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"))
    # Gemini settings
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    # Shared
    system_prompt: str = field(default_factory=lambda: PERSONAS.get(
        os.getenv("PERSONA", "narrator"), PERSONAS["narrator"]
    ))
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
    # Delivery tuning
    speed_factor: float = field(default_factory=lambda: float(os.getenv("TTS_SPEED", "1.0")))
    # Generation diversity — higher = more expressive but less stable
    temperature: float = field(default_factory=lambda: float(os.getenv("TTS_TEMPERATURE", "1.0")))
    top_p: float = field(default_factory=lambda: float(os.getenv("TTS_TOP_P", "1.0")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TTS_TOP_K", "15")))
    output_device: str | None = field(default_factory=lambda: os.getenv("AUDIO_OUTPUT_DEVICE", None))  # None=system default


@dataclass
class AppConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
