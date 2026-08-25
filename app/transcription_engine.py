from __future__ import annotations

import os
import threading
from pathlib import Path


class TranscriptionEngineError(RuntimeError):
    pass


class WhisperTranscriptionEngine:
    """Lazy local speech recognition powered by faster-whisper."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._device = self._select_device()
        self._model_name = os.getenv("WHISPER_MODEL", "small").strip() or "small"

    @staticmethod
    def _select_device() -> str:
        configured = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
        if configured != "auto":
            return configured
        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except (ImportError, RuntimeError):
            return "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionEngineError(
                "本地语音识别尚未安装。请执行 pip install -r requirements-tts.txt。"
            ) from exc

        compute_type = "int8_float16" if self._device == "cuda" else "int8"
        try:
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=compute_type,
            )
        except Exception as exc:
            if self._device != "cuda":
                raise TranscriptionEngineError(f"Whisper 模型加载失败：{exc}") from exc
            try:
                self._device = "cpu"
                self._model = WhisperModel(
                    self._model_name,
                    device="cpu",
                    compute_type="int8",
                )
            except Exception as cpu_exc:
                raise TranscriptionEngineError(
                    f"Whisper 的 CUDA 与 CPU 加载均失败：{cpu_exc}"
                ) from cpu_exc
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        with self._lock:
            model = self._load()
            try:
                segments, _ = model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = "".join(segment.text for segment in segments).strip()
            except Exception as exc:
                raise TranscriptionEngineError(f"本地语音识别失败：{exc}") from exc
        if not text:
            raise TranscriptionEngineError("没有识别到清晰语音，请靠近麦克风后重试")
        return text
