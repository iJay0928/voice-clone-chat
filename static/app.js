const $ = (selector) => document.querySelector(selector);
let referenceBlob = null;
let profileId = localStorage.getItem("voiceProfileId");
let conversationId = null;
let activeRecorder = null;

class WavRecorder {
  constructor(onTick, maxSeconds = null) { this.onTick = onTick; this.maxSeconds = maxSeconds; this.chunks = []; }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    this.context = new AudioContext();
    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (event) => this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    this.source.connect(this.processor); this.processor.connect(this.context.destination);
    this.startedAt = Date.now();
    this.timer = setInterval(() => {
      const elapsed = (Date.now() - this.startedAt) / 1000;
      this.onTick?.(elapsed);
      if (this.maxSeconds && elapsed >= this.maxSeconds) this.stop().then(this.onAutoStop);
    }, 100);
  }
  async stop() {
    if (!this.context) return null;
    clearInterval(this.timer); this.processor.disconnect(); this.source.disconnect(); this.stream.getTracks().forEach(t => t.stop());
    const sampleRate = this.context.sampleRate; await this.context.close(); this.context = null;
    const length = this.chunks.reduce((n, c) => n + c.length, 0); const samples = new Float32Array(length);
    let offset = 0; this.chunks.forEach(c => { samples.set(c, offset); offset += c.length; });
    return encodeWav(samples, sampleRate);
  }
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
  const str = (o, s) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); str(36, "data"); view.setUint32(40, samples.length * 2, true);
  let o = 44; for (const sample of samples) { const v = Math.max(-1, Math.min(1, sample)); view.setInt16(o, v < 0 ? v * 0x8000 : v * 0x7fff, true); o += 2; }
  return new Blob([view], { type: "audio/wav" });
}

function formatTime(seconds) { return `00:${String(Math.floor(seconds)).padStart(2, "0")}`; }
function refreshCreateState() { $("#create-profile").disabled = !(referenceBlob && $("#consent").checked); }
function setMessage(el, text, error = false) { el.textContent = text; el.style.color = error ? "#e06c3f" : "#216455"; }

async function request(url, options) {
  const response = await fetch(url, options); const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`); return data;
}

async function checkHealth() {
  try { const data = await request("/api/health"); $("#health").textContent = `服务在线 · XTTS ${data.voice_device.toUpperCase()} / WHISPER ${data.whisper_device.toUpperCase()}`; $("#health").classList.add("ok"); }
  catch { $("#health").textContent = "服务未连接"; }
}

$("#record-reference").addEventListener("click", async () => {
  const button = $("#record-reference");
  if (activeRecorder) { referenceBlob = await activeRecorder.stop(); activeRecorder = null; button.classList.remove("recording"); button.querySelector("b").textContent = "重新录制"; refreshCreateState(); return; }
  try {
    const rec = new WavRecorder(sec => { $("#reference-time").textContent = `${formatTime(sec)} / 00:15`; $("#reference-progress").style.width = `${Math.min(100, sec / 15 * 100)}%`; }, 15);
    rec.onAutoStop = blob => { referenceBlob = blob; activeRecorder = null; button.classList.remove("recording"); button.querySelector("b").textContent = "重新录制"; refreshCreateState(); };
    activeRecorder = rec; await rec.start(); button.classList.add("recording"); button.querySelector("b").textContent = "停止录制";
  } catch (e) { setMessage($("#setup-message"), `无法使用麦克风：${e.message}`, true); }
});

$("#reference-file").addEventListener("change", e => { referenceBlob = e.target.files[0] || null; $("#record-reference b").textContent = referenceBlob ? `已选择 ${referenceBlob.name}` : "开始录制"; refreshCreateState(); });
$("#consent").addEventListener("change", refreshCreateState);

$("#create-profile").addEventListener("click", async () => {
  const button = $("#create-profile"); button.disabled = true; setMessage($("#setup-message"), "正在验证参考音频…");
  const body = new FormData(); body.append("audio", referenceBlob, referenceBlob.name || "reference.wav"); body.append("consent", "true");
  try { const data = await request("/api/voice-profile", { method: "POST", body }); profileId = data.profile_id; localStorage.setItem("voiceProfileId", profileId); showStudio(data.duration); }
  catch (e) { setMessage($("#setup-message"), e.message, true); button.disabled = false; }
});

function showStudio(duration) { $("#setup").classList.add("hidden"); $("#studio").classList.remove("hidden"); if (duration) $("#profile-meta").textContent = `参考音频 ${duration.toFixed(1)} 秒`; }
if (profileId) showStudio();

$("#reset-profile").addEventListener("click", () => { localStorage.removeItem("voiceProfileId"); profileId = null; location.reload(); });
$("#clear-chat").addEventListener("click", () => { conversationId = null; $("#messages").innerHTML = '<div class="empty-state"><span>⌁</span><p>写下一句话，或点击麦克风开始说话</p></div>'; });

function addMessage(role, text, audioUrl) {
  $("#messages .empty-state")?.remove(); const div = document.createElement("div"); div.className = `bubble ${role}`; div.textContent = text;
  if (audioUrl) { const audio = document.createElement("audio"); audio.controls = true; audio.autoplay = true; audio.src = audioUrl; div.append(audio); }
  $("#messages").append(div); $("#messages").scrollTop = $("#messages").scrollHeight;
}

async function sendChat({ text = "", audioBlob = null }) {
  const status = $("#chat-status"), send = $("#send-button"); send.disabled = true; setMessage(status, audioBlob ? "正在识别语音…" : "正在思考并生成语音…");
  const body = new FormData(); body.append("profile_id", profileId); body.append("language", $("#language").value); if (text) body.append("text", text); if (audioBlob) body.append("audio", audioBlob, "message.wav"); if (conversationId) body.append("conversation_id", conversationId);
  try { const data = await request("/api/chat", { method: "POST", body }); addMessage("user", data.user_text); addMessage("assistant", data.answer, data.audio_url); conversationId = data.conversation_id; setMessage(status, ""); }
  catch (e) { setMessage(status, e.message, true); } finally { send.disabled = false; }
}

$("#chat-form").addEventListener("submit", e => { e.preventDefault(); const input = $("#chat-input"), text = input.value.trim(); if (!text) return; input.value = ""; sendChat({ text }); });
$("#chat-input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); } });

$("#mic-button").addEventListener("click", async () => {
  const button = $("#mic-button");
  if (activeRecorder) { const blob = await activeRecorder.stop(); activeRecorder = null; button.classList.remove("recording"); setMessage($("#chat-status"), ""); if (blob) sendChat({ audioBlob: blob }); return; }
  try { activeRecorder = new WavRecorder(sec => setMessage($("#chat-status"), `正在录音 ${formatTime(sec)}，再次点击结束`), 60); activeRecorder.onAutoStop = blob => { activeRecorder = null; button.classList.remove("recording"); sendChat({ audioBlob: blob }); }; await activeRecorder.start(); button.classList.add("recording"); }
  catch (e) { setMessage($("#chat-status"), `无法使用麦克风：${e.message}`, true); }
});

checkHealth();
