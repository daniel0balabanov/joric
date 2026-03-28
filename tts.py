"""GPT-SoVITS v2 TTS client.

Requires a running GPT-SoVITS v2 inference server:
    cd ~/GPT-SoVITS && .venv/bin/python api_v2.py -a 0.0.0.0 -p 9880
"""

import io
import logging

import numpy as np

from config import TTSConfig

log = logging.getLogger(__name__)


class TTS:
    """Synchronous TTS via GPT-SoVITS v2 HTTP API."""

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """POST text to /tts. Returns (audio_np, sample_rate) or None on error."""
        if not self._cfg.enabled:
            return None
        if not self._cfg.ref_audio_path:
            log.warning("TTS skipped: ref_audio_path not set")
            return None

        try:
            import httpx
            import soundfile as sf

            payload = {
                "text": text,
                "text_lang": self._cfg.text_lang,
                "prompt_lang": self._cfg.prompt_lang,
                "prompt_text": self._cfg.prompt_text,
                "ref_audio_path": self._cfg.ref_audio_path,
                "media_type": "wav",
                "streaming_mode": False,
            }
            resp = httpx.post(f"{self._cfg.api_url}/tts", json=payload, timeout=60.0)
            if not resp.is_success:
                log.warning("GPT-SoVITS error %d: %s", resp.status_code, resp.text[:200])
                return None

            audio_np, sample_rate = sf.read(io.BytesIO(resp.content))
            return audio_np, sample_rate
        except Exception as exc:
            log.warning("TTS error: %s", exc)
            return None

    def speak(self, text: str) -> None:
        """Synthesize and play *text*, blocking until playback finishes."""
        result = self.synthesize(text)
        if result is None:
            return
        audio_np, sample_rate = result
        try:
            import sounddevice as sd
            sd.play(audio_np, samplerate=sample_rate)
            sd.wait()
        except Exception as exc:
            log.warning("TTS playback error: %s", exc)
