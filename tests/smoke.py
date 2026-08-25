from __future__ import annotations

import io
import shutil
import struct
import sys
import wave
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as main

PROFILE_DIR = main.PROFILE_DIR
app = main.app


def wav_bytes(seconds: float, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<h", 0) * int(seconds * sample_rate))
    return buffer.getvalue()


client = TestClient(app)

health = client.get("/api/health")
assert health.status_code == 200 and health.json()["ok"] is True

index = client.get("/")
assert index.status_code == 200 and "声迹" in index.text

too_short = client.post(
    "/api/voice-profile",
    files={"audio": ("short.wav", wav_bytes(1), "audio/wav")},
    data={"consent": "true"},
)
assert too_short.status_code == 400

created = client.post(
    "/api/voice-profile",
    files={"audio": ("reference.wav", wav_bytes(6), "audio/wav")},
    data={"consent": "true"},
)
assert created.status_code == 200, created.text
profile_id = created.json()["profile_id"]
assert (PROFILE_DIR / profile_id / "reference.wav").is_file()

try:
    speak = client.post(
        "/api/speak",
        json={"profile_id": profile_id, "text": "你好", "language": "zh-cn"},
    )
    assert speak.status_code == 503  # model use stays gated until license acceptance

    captured_messages = []
    original_answer = main._deepseek_answer
    original_synthesize = main._synthesize
    original_transcribe = main.transcription_engine.transcribe

    def fake_answer(messages):
        captured_messages.append(messages)
        return "这是模拟回复。"

    main._deepseek_answer = fake_answer
    main._synthesize = lambda *_args: "/generated/mock.wav"
    main.transcription_engine.transcribe = lambda *_args: "这是本地识别结果"

    first = client.post(
        "/api/chat",
        data={"profile_id": profile_id, "language": "zh-cn", "text": "第一轮问题"},
    )
    assert first.status_code == 200, first.text
    conversation_id = first.json()["conversation_id"]

    second = client.post(
        "/api/chat",
        data={
            "profile_id": profile_id,
            "language": "zh-cn",
            "text": "第二轮问题",
            "conversation_id": conversation_id,
        },
    )
    assert second.status_code == 200, second.text
    assert [message["role"] for message in captured_messages[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    voice_chat = client.post(
        "/api/chat",
        files={"audio": ("message.wav", wav_bytes(1), "audio/wav")},
        data={"profile_id": profile_id, "language": "zh-cn"},
    )
    assert voice_chat.status_code == 200, voice_chat.text
    assert voice_chat.json()["user_text"] == "这是本地识别结果"
finally:
    if "original_answer" in locals():
        main._deepseek_answer = original_answer
        main._synthesize = original_synthesize
        main.transcription_engine.transcribe = original_transcribe
    target = (PROFILE_DIR / profile_id).resolve()
    assert PROFILE_DIR.resolve() in target.parents
    shutil.rmtree(target)

print("smoke-ok")
