import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2


def serve(assistant, host="127.0.0.1", port=8765, model_client=None):
    web = Path(__file__).resolve().parents[1] / "web" / "index.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    self.send(200, web.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self.send(200, assistant.snapshot())
                elif path == "/api/frame.jpg":
                    frame = assistant.preview()
                    height, width = frame.shape[:2]
                    half = assistant.config["center_fraction"] / 2
                    cv2.rectangle(frame, (int((0.5-half)*width), int((0.5-half)*height)),
                                  (int((0.5+half)*width), int((0.5+half)*height)), (130, 130, 130), 1)
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise RuntimeError("图像编码失败")
                    self.send(200, encoded.tobytes(), "image/jpeg")
                else:
                    self.send(404, {"error": "接口不存在"})
            except Exception as error:
                self.send(500, {"error": str(error)})

        def do_POST(self):
            path = urlsplit(self.path).path
            try:
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    self.send(403, {"error": "只允许本页面发起请求"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                if length < 0 or length > 10 * 1024 * 1024:
                    self.send(413, {"error": "请求过大"})
                    return
                if path == "/api/stop":
                    assistant.stop()
                    self.send(200, assistant.snapshot())
                    return
                body = self.rfile.read(length)
                if path == "/api/audio":
                    if model_client is None:
                        raise ValueError("语音转录需要启用模型并配置 JARVIS_ASR_MODEL")
                    if not body.startswith(b"RIFF") or body[8:12] != b"WAVE":
                        raise ValueError("请上传 WAV 录音")
                    self.send(200, assistant.submit_audio(body, model_client))
                    return
                data = json.loads(body or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("请求必须是 JSON 对象")
                if path == "/api/message":
                    if not isinstance(data.get("text"), str) or len(data["text"]) > 2000:
                        raise ValueError("目标描述必须是 2000 字以内的文字")
                    state = assistant.submit(data["text"])
                elif path == "/api/reset":
                    with assistant.lock:
                        assistant.reset()
                        if assistant.mode == "simulation":
                            assistant.camera.reset()
                    state = assistant.snapshot()
                elif path == "/api/scenario":
                    if assistant.mode != "simulation":
                        raise ValueError("故障场景只用于软件模拟")
                    name = data.get("name", "default")
                    if name not in assistant.camera.names:
                        raise ValueError("未知模拟场景")
                    with assistant.lock:
                        assistant.reset()
                        assistant.camera.reset(name)
                    state = assistant.snapshot()
                else:
                    self.send(404, {"error": "接口不存在"})
                    return
                self.send(200, state)
            except (ValueError, KeyError) as error:
                self.send(400, {"error": str(error)})
            except RuntimeError as error:
                self.send(409, {"error": str(error)})
            except Exception as error:
                self.send(500, {"error": str(error)})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"打开 http://{host}:{server.server_port}；Ctrl+C 停止。", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        assistant.close()
