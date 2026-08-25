# 本地音色克隆 AI 对话

输入约 15 秒参考音频，使用本地 XTTS v2 零样本音色克隆生成任意文本语音。聊天链路为：本地 faster-whisper 语音识别 → DeepSeek API 多轮回答 → 本地克隆音色朗读。

## 功能

- 浏览器直接录制 15 秒参考音频，或上传常见音频格式
- 本地 XTTS v2 音色克隆，参考音频不会上传到第三方
- 文本聊天与麦克风语音聊天
- faster-whisper 本地语音识别
- DeepSeek Chat Completions 多轮上下文
- 明确授权确认、文件类型/大小/时长校验

注意：仅克隆你本人的声音，或已获得声音所有者明确授权的声音。请不要用于冒充、欺诈、骚扰或绕过声纹认证。

## 环境要求

- Windows / macOS / Linux
- Python 3.10 或 3.11（推荐 3.11）
- NVIDIA GPU + 8 GB 左右显存体验更好；CPU 也可运行但合成较慢
- 非 WAV 上传需要系统已安装 `ffmpeg`
- DeepSeek API Key

## 安装

```powershell
cd D:\voice-clone-chat
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-tts.txt
Copy-Item .env.example .env
```

若使用 NVIDIA GPU，建议先按 PyTorch 官网与你的 CUDA 版本匹配的命令安装 `torch` 和 `torchaudio`，再安装其余依赖。

编辑 `.env`，填入 `DEEPSEEK_API_KEY`。默认使用低延迟的 `deepseek-v4-flash`；可以通过 `DEEPSEEK_MODEL` 切换其他已开通模型。

首次使用麦克风时，faster-whisper 会下载 `small` 模型；可通过 `WHISPER_MODEL` 改成 `medium` 提高准确率。音频识别完全在本机完成，只有转写后的文字与最近对话历史会发送给 DeepSeek。

默认优先使用 CUDA；如果 Windows 环境缺少 CTranslate2 所需的 GPU 动态库，会自动回退到 CPU 识别。

首次合成前请阅读 Coqui Public Model License；同意后将 `COQUI_TOS_AGREED` 改为 `1`。首次朗读会下载 XTTS v2 模型，可能需要数分钟。

## 启动

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000>。

## API

- `POST /api/voice-profile`：上传并验证参考音频
- `POST /api/speak`：将任意文本用指定音色朗读
- `POST /api/chat`：文本或本地语音识别 → DeepSeek → 克隆音色回答
- `GET /api/health`：检查配置和推理设备
- `GET /docs`：交互式 API 文档

音色文件保存在 `storage/profiles`，生成结果保存在 `storage/generated`。删除对应目录即可清除本地数据。
