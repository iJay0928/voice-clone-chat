from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .transcription_engine import TranscriptionEngineError, WhisperTranscriptionEngine
from .voice_engine import VoiceEngineError, XTTSVoiceEngine

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
PROFILE_DIR = BASE_DIR / "storage" / "profiles"
GENERATED_DIR = BASE_DIR / "storage" / "generated"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PROFILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SUPPORTED_LANGUAGES = {"zh-cn", "en", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"}
SYSTEM_MESSAGE = (
    "你正在进行自然的语音对话。回答要口语化、简洁、温暖，通常不超过三句话。"
    "不要声称自己就是被克隆声音的本人，也不要假装拥有那个人的记忆。"
)
MAX_CONVERSATIONS = 100
MAX_HISTORY_MESSAGES = 12

load_dotenv(BASE_DIR / ".env")
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Voice Clone Chat", version="1.0.0")
voice_engine = XTTSVoiceEngine()
transcription_engine = WhisperTranscriptionEngine()
conversations: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
conversations_lock = threading.Lock()


class SpeakRequest(BaseModel):
    profile_id: str
    text: str = Field(min_length=1, max_length=1000)
    language: str = "zh-cn"


def _profile_wav(profile_id: str) -> Path:
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise HTTPException(400, "无效的音色 ID")
    path = PROFILE_DIR / profile_id / "reference.wav"
    if not path.is_file():
        raise HTTPException(404, "找不到该音色，请重新创建")
    return path


def _language(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"暂不支持语言：{value}")
    return normalized


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                target.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "音频文件不能超过 25 MB")
            target.write(chunk)


def _convert_to_wav(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".wav":
        shutil.move(str(source), str(destination))
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(400, "非 WAV 音频需要安装 ffmpeg；也可以直接上传 WAV 文件")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-ac", "1", "-ar", "24000", str(destination)],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    source.unlink(missing_ok=True)
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "无法解析该音频文件")


def _validate_reference(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            duration = wav.getnframes() / float(wav.getframerate())
            if wav.getnchannels() not in (1, 2) or wav.getsampwidth() != 2:
                raise HTTPException(400, "请使用 16-bit PCM WAV 音频")
    except (wave.Error, EOFError, ZeroDivisionError) as exc:
        raise HTTPException(400, "WAV 文件无效或已损坏") from exc
    if not 5 <= duration <= 30:
        raise HTTPException(400, "参考音频时长应在 5 到 30 秒之间，推荐约 15 秒")
    return round(duration, 1)


def _deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(503, "尚未配置 DEEPSEEK_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def _conversation_messages(
    conversation_id: str | None, user_text: str
) -> tuple[str, list[dict[str, str]]]:
    if conversation_id and not PROFILE_ID_RE.fullmatch(conversation_id):
        raise HTTPException(400, "无效的对话 ID")
    with conversations_lock:
        if not conversation_id or conversation_id not in conversations:
            conversation_id = uuid.uuid4().hex
            conversations[conversation_id] = []
        history = conversations[conversation_id][-MAX_HISTORY_MESSAGES:]
        conversations.move_to_end(conversation_id)
        while len(conversations) > MAX_CONVERSATIONS:
            conversations.popitem(last=False)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        *history,
        {"role": "user", "content": user_text},
    ]
    return conversation_id, messages


def _remember_turn(conversation_id: str, user_text: str, answer: str) -> None:
    with conversations_lock:
        history = conversations.setdefault(conversation_id, [])
        history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]
        )
        del history[:-MAX_HISTORY_MESSAGES]


def _deepseek_answer(messages: list[dict[str, str]]) -> str:
    try:
        response = _deepseek_client().chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            messages=messages,
            max_tokens=350,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"DeepSeek 请求失败：{exc}") from exc
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise HTTPException(502, "DeepSeek 未返回可朗读内容")
    return answer


def _synthesize(profile_id: str, text: str, language: str) -> str:
    output_name = f"{uuid.uuid4().hex}.wav"
    try:
        voice_engine.synthesize(
            text=text,
            reference_wav=_profile_wav(profile_id),
            language=_language(language),
            output_wav=GENERATED_DIR / output_name,
        )
    except VoiceEngineError as exc:
        raise HTTPException(503, str(exc)) from exc
    return f"/generated/{output_name}"


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "voice_engine": "ready" if voice_engine.loaded else "lazy",
        "voice_device": voice_engine.device,
        "whisper_engine": "ready" if transcription_engine.loaded else "lazy",
        "whisper_device": transcription_engine.device,
        "whisper_model": transcription_engine.model_name,
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


@app.post("/api/voice-profile")
async def create_voice_profile(
    audio: Annotated[UploadFile, File()],
    consent: Annotated[bool, Form()],
) -> dict:
    if not consent:
        raise HTTPException(400, "必须确认已获得声音所有者的明确授权")
    suffix = Path(audio.filename or "reference.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}:
        raise HTTPException(400, "不支持该音频格式")
    profile_id = uuid.uuid4().hex
    profile_folder = PROFILE_DIR / profile_id
    raw_path = profile_folder / f"upload{suffix}"
    wav_path = profile_folder / "reference.wav"
    try:
        await _save_upload(audio, raw_path)
        _convert_to_wav(raw_path, wav_path)
        duration = _validate_reference(wav_path)
    except Exception:
        shutil.rmtree(profile_folder, ignore_errors=True)
        raise
    return {"profile_id": profile_id, "duration": duration}


@app.post("/api/speak")
def speak(payload: SpeakRequest) -> dict:
    return {"audio_url": _synthesize(payload.profile_id, payload.text.strip(), payload.language)}


@app.post("/api/chat")
async def chat(
    profile_id: Annotated[str, Form()],
    language: Annotated[str, Form()] = "zh-cn",
    text: Annotated[str | None, Form()] = None,
    conversation_id: Annotated[str | None, Form()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
) -> dict:
    _profile_wav(profile_id)
    lang = _language(language)
    user_text = (text or "").strip()

    if audio is not None and audio.filename:
        suffix = Path(audio.filename).suffix.lower() or ".wav"
        if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}:
            raise HTTPException(400, "不支持该聊天音频格式")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            await _save_upload(audio, temp_path)
            whisper_language = "zh" if lang == "zh-cn" else lang
            user_text = await run_in_threadpool(
                transcription_engine.transcribe, temp_path, whisper_language
            )
        except TranscriptionEngineError as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)

    if not user_text:
        raise HTTPException(400, "请输入文字或录制一段语音")
    if len(user_text) > 4000:
        raise HTTPException(400, "单次消息不能超过 4000 字符")

    conversation_id, messages = _conversation_messages(conversation_id, user_text)
    answer = await run_in_threadpool(_deepseek_answer, messages)
    _remember_turn(conversation_id, user_text, answer)
    audio_url = await run_in_threadpool(_synthesize, profile_id, answer, lang)
    return {
        "user_text": user_text,
        "answer": answer,
        "conversation_id": conversation_id,
        "audio_url": audio_url,
    }


app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str = ""):
    candidate = STATIC_DIR / path
    if path and candidate.is_file() and STATIC_DIR in candidate.resolve().parents:
        return FileResponse(candidate)
    return FileResponse(STATIC_DIR / "index.html")
