"""Speech-to-text using faster-whisper large-v3-turbo."""

import logging

import numpy as np

from config import STTConfig

log = logging.getLogger(__name__)


class Transcriber:
    """Wraps faster-whisper for speech-to-text transcription."""

    def __init__(self, cfg: STTConfig) -> None:
        self._cfg = cfg
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        compute_type = self._cfg.compute_type
        device = self._cfg.device

        try:
            self._model = WhisperModel(
                self._cfg.model_size,
                device=device,
                compute_type=compute_type,
            )
            log.info("Whisper %s loaded on %s (%s)", self._cfg.model_size, device, compute_type)
        except Exception:
            log.warning("Failed to load on %s/%s, falling back to cpu/int8", device, compute_type)
            self._model = WhisperModel(
                self._cfg.model_size,
                device="cpu",
                compute_type="int8",
            )
            log.info("Whisper %s loaded on cpu (int8)", self._cfg.model_size)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 16 kHz numpy array. Returns stripped text."""
        self._load()

        if audio is None or len(audio) == 0:
            return ""

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._cfg.language,
                beam_size=self._cfg.beam_size,
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()
            log.debug("Transcribed [lang=%s, prob=%.2f]: %s", info.language, info.language_probability, text)
            return text
        except RuntimeError as exc:
            if "libcublas" in str(exc) or "CUDA" in str(exc):
                log.warning("CUDA inference failed (%s), reloading on cpu/int8", exc)
                self._model = None
                self._cfg.device = "cpu"
                self._cfg.compute_type = "int8"
                self._load()
                segments, info = self._model.transcribe(
                    audio,
                    language=self._cfg.language,
                    beam_size=self._cfg.beam_size,
                    vad_filter=True,
                )
                text = " ".join(seg.text for seg in segments).strip()
                return text
            raise
