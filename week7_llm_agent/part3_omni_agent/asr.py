"""Speech-to-text adapters used by the visual/voice agent."""

from __future__ import annotations

import io
import os


class SiliconFlowASR:
    """Transcribe WAV bytes with SiliconFlow's OpenAI-compatible ASR API."""

    def __init__(self, *, model: str = "FunAudioLLM/SenseVoiceSmall",
                 base_url: str = "https://api.siliconflow.cn/v1",
                 api_key_env: str = "SILICONFLOW_API_KEY", client=None):
        self.model = model
        if client is None:
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"请先设置环境变量 {api_key_env}（ASR API Key 不写进代码或仓库）")
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=base_url, timeout=60)
        self.client = client

    def transcribe(self, wav: bytes) -> str:
        audio = io.BytesIO(wav)
        audio.name = "speech.wav"
        response = self.client.audio.transcriptions.create(model=self.model, file=audio)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("语音识别没有返回有效文字")
        return text.strip()
