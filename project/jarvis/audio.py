import io
import os
import subprocess
import time
import wave

import numpy as np


def record_push_to_talk(*, device=None, max_seconds=30, sample_rate=16000, start_prompt=True):
    """Press Enter to start/finish; retain at most max_seconds without input threads."""
    import sounddevice as sd

    if start_prompt:
        input("按回车开始录音，录音后再按回车结束：")
    chunks = []
    started = time.monotonic()
    max_samples = int(max_seconds * sample_rate)
    recorded = 0

    def callback(data, frames, timing, status):
        nonlocal recorded
        remaining = max_samples - recorded
        if remaining > 0:
            chunk = data[:remaining].copy()
            chunks.append(chunk)
            recorded += len(chunk)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        device=device, callback=callback):
        input(f"录音中（最多保留 {max_seconds:g} 秒），说完按回车：")
    if time.monotonic() - started < 0.25 or recorded < sample_rate // 4:
        raise ValueError("录音不足0.25秒，请重新录音")
    samples = np.concatenate(chunks).reshape(-1)
    if int(np.abs(samples.astype(np.int32)).max()) < 100:
        raise ValueError("未检测到有效声音，请检查麦克风")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    return buffer.getvalue()


def speak(text):
    if not text.strip():
        return
    if os.name == "nt":
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$speech = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "$speech.Speak($env:JARVIS_TTS_TEXT)")
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       env={**os.environ, "JARVIS_TTS_TEXT": text[:1000]}, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
    else:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
