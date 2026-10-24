"""Microphone recording and speech output for the omni agent."""

from __future__ import annotations

import base64
import io
import os
import queue
import subprocess
import threading
import wave

import numpy as np

SAMPLE_RATE = 16000
OMNI_AUDIO_RATE = 24000  # Qwen-Omni returns 24 kHz, 16-bit mono PCM


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buffer.getvalue()


def audio_data_url(wav: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def record(*, seconds: float | None = None, device=None, max_seconds: float = 30.0) -> bytes:
    """Record from the microphone: fixed length, or until Enter is pressed (push-to-talk)."""
    import sounddevice as sd

    chunks: queue.Queue = queue.Queue()
    stop = threading.Event()

    def callback(data, frames, timestamp, status):
        chunks.put(data.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device, callback=callback):
        if seconds is not None:
            stop.wait(min(seconds, max_seconds))
        else:
            waiter = threading.Thread(target=lambda: (input(), stop.set()), daemon=True)
            waiter.start()
            if not stop.wait(max_seconds):
                print(f"（已达到 {max_seconds:g} 秒上限，自动结束录音）")
    parts = []
    while not chunks.empty():
        parts.append(chunks.get())
    samples = np.concatenate(parts).reshape(-1) if parts else np.zeros(0, np.int16)
    if len(samples) < SAMPLE_RATE * 0.3:
        raise ValueError("录音太短，请按回车后说话，说完再按回车")
    peak = int(np.max(np.abs(samples.astype(np.int32))))
    if peak < 300:
        raise ValueError("几乎没有录到声音，请检查麦克风或 Windows 麦克风隐私权限")
    return wav_bytes(samples)


def play_pcm(chunks_b64: list[str], rate: int = OMNI_AUDIO_RATE):
    """Play base64 PCM/WAV chunks returned by an omni model with audio output."""
    import sounddevice as sd

    raw = b"".join(base64.b64decode(chunk) for chunk in chunks_b64)
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw)) as file:
            rate = file.getframerate()
            raw = file.readframes(file.getnframes())
    if raw:
        sd.play(np.frombuffer(raw, dtype=np.int16), rate)
        sd.wait()


def speak(text: str):
    """Local text-to-speech: pyttsx3 if installed, otherwise Windows System.Speech."""
    text = (text or "").strip()
    if not text:
        return
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    except ImportError:
        pass
    if os.name == "nt":
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak($env:AGENT_TTS_TEXT)")
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False,
                       env={**os.environ, "AGENT_TTS_TEXT": text[:500]})
