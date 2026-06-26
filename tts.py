"""TTS backends: GPT-SoVITS v2 (HTTP server) and OmniVoice (local model).

Select via TTS_BACKEND env var: "gptsovits" (default) or "omnivoice".
"""

import io
import logging
import re

import numpy as np

from config import TTSConfig

log = logging.getLogger(__name__)


class TTS:
    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._omnivoice = None  # lazy-loaded

    @staticmethod
    def _split_chunks(text: str) -> list[str]:
        import re
        parts = re.split(r'(?<=\.)\s+', text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _normalize(text: str) -> str:
        """Convert ALL CAPS words to lowercase so TTS reads them as words, not acronyms."""
        return re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group(1).lower(), text)

    @staticmethod
    def _clean_punctuation(text: str) -> str:
        """Remove all punctuation — OmniVoice vocalizes it as audio artifacts."""
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """Return (audio_np, sample_rate) or None on error/disabled."""
        if not self._cfg.enabled:
            return None
        text = self._normalize(text)
        if self._cfg.backend == "omnivoice":
            chunks = [self._clean_punctuation(c) for c in (self._split_chunks(text) or [text])]
            if len(chunks) == 1:
                return self._synthesize_omnivoice(chunks[0])
            parts = []
            for i, chunk in enumerate(chunks, 1):
                log.info("Chunk %d/%d: %s", i, len(chunks), chunk[:60])
                result = self._synthesize_omnivoice(chunk)
                if result is not None:
                    parts.append(result[0])
                    sr = result[1]
            if not parts:
                return None
            return np.concatenate(parts), sr
        return self._synthesize_gptsovits(text)

    def synthesize_mp3(self, text: str) -> bytes | None:
        """Return MP3-encoded audio bytes, or None on error/disabled."""
        result = self.synthesize(text)
        if result is None:
            return None
        audio_np, sample_rate = result
        try:
            import soundfile as sf
            from pydub import AudioSegment

            wav_buf = io.BytesIO()
            sf.write(wav_buf, audio_np, sample_rate, format="wav")
            wav_buf.seek(0)

            seg = AudioSegment.from_file(wav_buf, format="wav")
            mp3_buf = io.BytesIO()
            seg.export(mp3_buf, format="mp3")
            return mp3_buf.getvalue()
        except Exception as exc:
            log.warning("MP3 encode error: %s", exc)
            return None

    def speak(self, text: str) -> None:
        """Synthesize and play text, blocking until playback finishes."""
        result = self.synthesize(text)
        if result is None:
            return
        audio_np, sample_rate = result
        try:
            import sounddevice as sd
            sd.play(audio_np, samplerate=sample_rate, device=self._cfg.output_device)
            sd.wait()
        except Exception as exc:
            log.warning("TTS playback error: %s", exc)

    # ------------------------------------------------------------------
    # GPT-SoVITS backend
    # ------------------------------------------------------------------

    def _synthesize_gptsovits(self, text: str) -> tuple[np.ndarray, int] | None:
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
                "speed_factor": self._cfg.speed_factor,
                "temperature": self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "top_k": self._cfg.top_k,
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

    # ------------------------------------------------------------------
    # OmniVoice backend
    # ------------------------------------------------------------------

    def _load_omnivoice(self) -> None:
        if self._omnivoice is not None:
            return
        import torch
        from omnivoice import OmniVoice

        log.info("Loading OmniVoice model %s on %s …", self._cfg.omnivoice_model, self._cfg.omnivoice_device)
        self._omnivoice = OmniVoice.from_pretrained(
            self._cfg.omnivoice_model,
            device_map=self._cfg.omnivoice_device,
            dtype=torch.float16,
        )
        log.info("OmniVoice ready.")

    def _load_ref_audio(self, paths: str | list[str]) -> tuple[np.ndarray, int]:
        """Load and concatenate one or more reference WAV files."""
        import soundfile as sf
        if isinstance(paths, str):
            audio, sr = sf.read(paths, dtype="float32")
            return audio, sr
        chunks = []
        sr = None
        for p in paths:
            a, s = sf.read(p, dtype="float32")
            if sr is None:
                sr = s
            elif s != sr:
                raise ValueError(f"Sample rate mismatch in ref clips: {s} vs {sr}")
            chunks.append(a)
        return np.concatenate(chunks), sr

    def _synthesize_omnivoice(self, text: str) -> tuple[np.ndarray, int] | None:
        try:
            self._load_omnivoice()
            import torch
            ref_np, sr = self._load_ref_audio(self._cfg.ref_audio_path)
            ref_tensor = torch.from_numpy(ref_np).unsqueeze(0)  # (1, T)
            from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
            gen_cfg = OmniVoiceGenerationConfig(
                position_temperature=self._cfg.omnivoice_position_temperature,
                class_temperature=self._cfg.omnivoice_class_temperature,
            )
            audio = self._omnivoice.generate(
                text=text,
                ref_audio=(ref_tensor, sr),
                ref_text=self._cfg.prompt_text,
                generation_config=gen_cfg,
            )
            audio_np = np.array(audio[0], dtype=np.float32)
            return audio_np, 24000
        except Exception as exc:
            log.warning("OmniVoice error: %s", exc)
            return None
