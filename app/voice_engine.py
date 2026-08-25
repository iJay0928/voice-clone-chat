from __future__ import annotations

import os
import threading
from pathlib import Path


class VoiceEngineError(RuntimeError):
    pass


def _install_xtts_wav_loader() -> None:
    """Read reference WAV files without TorchCodec.

    Newer torchaudio releases route ``torchaudio.load`` through TorchCodec,
    which requires a shared-library FFmpeg build on Windows. References are
    already normalized to WAV when profiles are created, so SoundFile is a
    smaller and more reliable decoder for this path.
    """
    import soundfile as sf
    import torch
    import torchaudio
    from TTS.tts.models import xtts as xtts_module

    def load_audio(audiopath, sampling_rate):
        samples, source_rate = sf.read(
            str(audiopath), dtype="float32", always_2d=True
        )
        audio = torch.from_numpy(samples.T.copy())
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if source_rate != sampling_rate:
            audio = torchaudio.functional.resample(
                audio, source_rate, sampling_rate
            )
        return audio.clip_(-1, 1)

    xtts_module.load_audio = load_audio


class XTTSVoiceEngine:
    """Lazy-loaded local XTTS v2 engine.

    Loading is deferred so the web UI and health endpoint remain usable before
    the large model has been downloaded. A lock protects the shared model from
    concurrent synthesis calls.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._device = self._select_device()

    @staticmethod
    def _select_device() -> str:
        configured = os.getenv("VOICE_DEVICE", "auto").strip().lower()
        if configured != "auto":
            return configured
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return self._model
        if os.getenv("COQUI_TOS_AGREED", "0") != "1":
            raise VoiceEngineError(
                "请先阅读并接受 Coqui Public Model License，然后在 .env 中设置 "
                "COQUI_TOS_AGREED=1。"
            )
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise VoiceEngineError(
                "本地语音引擎尚未安装。请执行 pip install -r requirements-tts.txt。"
            ) from exc

        try:
            _install_xtts_wav_loader()
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                self._device
            )
        except Exception as exc:  # model download/runtime errors vary by platform
            raise VoiceEngineError(f"XTTS v2 加载失败：{exc}") from exc
        return self._model

    def synthesize(
        self, *, text: str, reference_wav: Path, language: str, output_wav: Path
    ) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            model = self._load()
            try:
                model.tts_to_file(
                    text=text,
                    speaker_wav=str(reference_wav),
                    language=language,
                    file_path=str(output_wav),
                )
            except Exception as exc:
                output_wav.unlink(missing_ok=True)
                raise VoiceEngineError(f"语音生成失败：{exc}") from exc
