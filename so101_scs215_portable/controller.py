"""Portable local/LAN web controller for the six-servo SCS215 arm."""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import signal
import socket
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from servo_bus import ServoBus, WATCHDOG_TIMEOUT_SECONDS, load_config
from joint_mapping import mapped_position


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
PRESET_NAMES = ("立正", "姿态2", "姿态3")


def local_ipv4_addresses() -> list[str]:
    addresses = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0]
            if not address.startswith("127.") and not address.startswith("169.254."):
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


class RequestHandler(BaseHTTPRequestHandler):
    bus: ServoBus | None = None
    access_token = ""

    def log_message(self, fmt, *args):
        print(f"Web: {self.address_string()} - {fmt % args}")

    def _authorized(self) -> bool:
        if not self.access_token:
            return True
        query = parse_qs(urlparse(self.path).query)
        supplied = self.headers.get("X-Control-Token", "") or query.get("token", [""])[0]
        return secrets.compare_digest(supplied, self.access_token)

    def _send_json(self, value, status=HTTPStatus.OK):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path, content_type: str):
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 4096:
            raise ValueError("请求体长度无效")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _reject_if_unauthorized(self) -> bool:
        if self._authorized():
            return False
        self._send_json({"ok": False, "error": "访问令牌无效"}, HTTPStatus.FORBIDDEN)
        return True

    def do_GET(self):
        if self._reject_if_unauthorized():
            return
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/api/state":
                runner = self.server.pose_runner
                config = runner.arm.config if runner is not None else json.loads((ROOT / "advanced.json").read_text(encoding="utf-8"))
                servos = self.bus.state()
                for servo in servos:
                    mapping = config.get("joint_mapping", {}).get(str(servo["id"]))
                    if mapping is not None:
                        servo["mapped_position"] = mapped_position(servo["position"], mapping)
                        servo["mapped_limits"] = mapping["limits"]
                ready, blocked_reason = False, ""
                if config["motion_calibrated"] is True and config["kinematics"]["calibrated"] is True:
                    from advanced import Kinematics
                    model = runner.arm.model if runner is not None else Kinematics(config)
                    current = [servo["position"] for servo in sorted(servos, key=lambda servo: servo["id"])]
                    try:
                        model.inverse(model.forward(current), current, allow_crossing=False)
                        ready = all(servo["mode"] == "position" for servo in servos)
                        if not ready:
                            blocked_reason = "末端控制需要六关节处于位置模式"
                    except ValueError as error:
                        blocked_reason = str(error)
                self._send_json(
                    {
                        "ok": True,
                        "port": self.bus.port,
                        "servos": servos,
                        "poses": list(PRESET_NAMES),
                        "cartesian": {
                            "ready": ready,
                            "blocked_reason": blocked_reason,
                            "initialized": runner is not None and runner.arm.cartesian_initialized,
                            "calibrated": config["motion_calibrated"] is True and config["kinematics"]["calibrated"] is True,
                            "settings": {key: config["kinematics"].get(key, [0] * 5)
                                         for key in ("zero_raw", "reference_angles_deg", "directions", "degrees_per_raw")},
                            "max_distance_mm": config.get("max_distance_mm", config["max_step_mm"]),
                            "result": runner.arm.last_move if runner is not None else None,
                            "joint_move": runner.arm.last_joint_move if runner is not None else None,
                        },
                        "pose": {
                            "name": self.server.pose_name,
                            "kind": self.server.pose_kind,
                            "busy": runner is not None and runner.busy,
                            "holding": runner is not None and runner.arm.holding,
                            "error": str(runner.error) if runner is not None and runner.error else "",
                            "stop_reason": self.server.pose_stop_reason,
                        },
                    }
                )
            elif path == "/api/health":
                self._send_json({"ok": True, "port": self.bus.port})
            else:
                self._send_json({"ok": False, "error": "页面不存在"}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        if self._reject_if_unauthorized():
            return
        path = urlparse(self.path).path
        try:
            data = self._json_body()
            with self.server.motion_lock:
                runner = self.server.pose_runner
                if path in {"/api/pose", "/api/cartesian/home"} and runner is not None and runner.arm.holding:
                    runner.stop()
                    runner.thread.join(timeout=2)
                if path in {"/api/mode", "/api/run", "/api/pose", "/api/cartesian", "/api/cartesian/home", "/api/cartesian/settings"} and runner is not None and runner.busy:
                    raise ValueError("当前姿态仍在执行，请等待结束或点击停止")
                if path == "/api/cartesian/settings":
                    if self.bus.running_id is not None:
                        raise ValueError("请先停止当前关节运动")
                    if data.get("confirmed") is not True:
                        raise ValueError("请勾选本次手动测试确认")
                    from advanced import Arm, CommandRunner, Kinematics, load_advanced
                    candidate = copy.deepcopy(runner.arm.config) if runner is not None else load_advanced()
                    for key in ("zero_raw", "reference_angles_deg", "directions"):
                        values = data[key]
                        if (not isinstance(values, list) or len(values) != 5
                                or any(type(value) not in (int, float) for value in values)):
                            raise ValueError("每组参数应包含五个数值")
                        candidate["kinematics"][key] = values
                    if any(not 0 <= value <= 1023 or int(value) != value
                           for value in candidate["kinematics"]["zero_raw"]):
                        raise ValueError("参考读数应为 0～1023 的整数")
                    candidate["kinematics"]["degrees_per_raw"] = float(data["degrees_per_raw"])
                    candidate["kinematics"]["calibrated"] = False
                    model = Kinematics(candidate)
                    if runner is None:
                        runner = self.server.pose_runner = CommandRunner(Arm(candidate, self.bus))
                    else:
                        runner.arm.config = candidate
                        runner.arm.model = model
                    runner.arm.ready = False
                    runner.arm.cartesian_initialized = False
                    runner.arm.last_move = None
                    self.server.pose_name = None
                    self.server.pose_kind = "pose"
                    self.server.pose_stop_reason = ""
                    self._send_json({"ok": True})
                elif path in {"/api/pose", "/api/cartesian", "/api/cartesian/home"}:
                    if path == "/api/pose":
                        name = data["name"]
                        if name not in PRESET_NAMES:
                            raise ValueError("请选择立正、姿态2或姿态3")
                    if self.bus.running_id is not None:
                        raise ValueError("请先停止当前关节运动")
                    if runner is None:
                        from advanced import Arm, CommandRunner, load_advanced
                        runner = self.server.pose_runner = CommandRunner(Arm(load_advanced(), self.bus))
                    if path == "/api/cartesian":
                        if (runner.arm.config["motion_calibrated"] is not True
                                or runner.arm.config["kinematics"]["calibrated"] is not True):
                            raise ValueError("末端控制前需核对 URDF 零位、关节方向和角度换算，完成运动学标定")
                        directions = {"前": (0, 1), "后": (0, -1), "左": (1, 1), "右": (1, -1), "上": (2, 1), "下": (2, -1)}
                        direction = data["direction"]
                        if direction not in directions or type(data.get("continuous", False)) is not bool:
                            raise ValueError("请选择前、后、左、右、上或下，并使用布尔值 continuous")
                        continuous = data.get("continuous", False)
                        amount = float(runner.arm.config["step_mm"] if continuous else data["distance_mm"])
                        maximum = float(runner.arm.config["max_step_mm"] if continuous else runner.arm.config.get("max_distance_mm", runner.arm.config["max_step_mm"]))
                        if not 0 < amount <= maximum:
                            raise ValueError(f"距离应在 0～{maximum:g} mm 之间")
                        runner.arm.cartesian_start_pose()
                        axis, sign = directions[direction]
                        delta = [0.0] * 3
                        delta[axis] = sign * amount
                        command = ("jog" if continuous else "move", delta)
                        name = f"末端向{direction}" + ("持续移动" if continuous else f"移动 {amount:g} mm")
                    else:
                        if path == "/api/cartesian/home":
                            name = "初始状态"
                            runner.arm.cartesian_initialized = False
                        runner.arm.validate_pose(runner.arm.config["poses"][name])
                        command = ("home" if path == "/api/cartesian/home" else "pose", name)
                    runner.arm.ready = False
                    runner.arm.last_move = None
                    self.server.pose_name = name
                    self.server.pose_kind = "cartesian" if path.startswith("/api/cartesian") else "pose"
                    self.server.pose_stop_reason = ""
                    self.server.pose_heartbeat = time.monotonic()
                    if not runner.submit(command):
                        raise ValueError("当前动作尚未结束")
                    self._send_json({"ok": True, "name": name})
                elif path == "/api/mode":
                    if runner is not None:
                        runner.arm.ready = False
                    result = self.bus.set_mode(data.get("id", "all"), data["mode"])
                    self._send_json({"ok": True, "result": result})
                elif path == "/api/run":
                    if runner is not None:
                        runner.arm.ready = False
                    result = self.bus.run(data["id"], data["direction"], data["speed"])
                    self._send_json({"ok": True, "run": result})
                elif path == "/api/heartbeat":
                    if data["id"] == "pose":
                        self.server.pose_heartbeat = time.monotonic()
                        result = {"id": "pose"}
                    else:
                        result = self.bus.heartbeat(data["id"])
                    self._send_json({"ok": True, "heartbeat": result})
                elif path == "/api/stop":
                    selector = "all" if runner is not None and runner.busy else data.get("id", "all")
                    if runner is not None:
                        runner.arm.cancel.set()
                        runner.arm.ready = False
                    failures = self.bus.stop(selector)
                    if runner is not None:
                        self.server.pose_stop_reason = ("停止未确认：" + "；".join(failures)
                                                        if failures else "关扭矩指令已发送")
                    self._send_json(
                        {"ok": not failures, "failures": failures},
                        HTTPStatus.OK if not failures else HTTPStatus.BAD_GATEWAY,
                    )
                else:
                    self._send_json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


class ControlHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self.pose_runner = None
        self.pose_name = None
        self.pose_kind = "pose"
        self.pose_stop_reason = ""
        self.pose_heartbeat = 0.0
        self.motion_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def service_actions(self):
        with self.motion_lock:
            if (self.pose_runner is not None and self.pose_runner.busy
                    and time.monotonic() - self.pose_heartbeat > WATCHDOG_TIMEOUT_SECONDS):
                self.pose_stop_reason = "网页控制心跳中断，动作已停止"
                try:
                    self.pose_runner.stop()
                except Exception as error:
                    self.pose_stop_reason = f"停止失败：{error}"
                    print(self.pose_stop_reason, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="SCS215 六关节连续旋转网页控制器")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--port", help="覆盖 config.json 中的串口，例如 COM8")
    parser.add_argument("--http-port", type=int, help="覆盖网页端口")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--lan", action="store_true", help="监听局域网并启用访问令牌")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--cartesian-view", action="store_true", help="打开末端六方向键盘展示界面")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    host = "0.0.0.0" if args.lan else args.host
    http_port = args.http_port or int(config.get("http_port", 8767))
    token = secrets.token_urlsafe(18) if args.lan else ""

    bus = ServoBus(config, args.port)
    RequestHandler.bus = bus
    RequestHandler.access_token = token
    try:
        server = ControlHTTPServer((host, http_port), RequestHandler)
    except Exception:
        bus.close()
        raise

    parameters = {"token": token} if token else {}
    if args.cartesian_view:
        parameters["view"] = "cartesian"
    query = "?" + urlencode(parameters) if parameters else ""
    local_url = f"http://127.0.0.1:{http_port}/{query}"

    def request_shutdown(_signum=None, _frame=None):
        # shutdown() cannot be called from the serve_forever thread itself.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)

    print("=" * 64)
    print(f"Servo bus: {bus.port} @ {config['baud_rate']} baud")
    print(f"Local control: {local_url}")
    if args.lan:
        addresses = local_ipv4_addresses()
        if addresses:
            for address in addresses:
                print(f"LAN control:   http://{address}:{http_port}/{query}")
        else:
            print("LAN control:   请查询本机 IPv4 地址后使用上面的端口和令牌")
        print("Only share the complete tokenized URL with the operator.")
    print("Press Ctrl+C to stop all motors, release COM port, and exit.")
    print("=" * 64)

    try:
        if not args.no_browser:
            threading.Timer(0.7, lambda: webbrowser.open(local_url)).start()
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        try:
            if server.pose_runner is not None:
                server.pose_runner.close()
        finally:
            bus.close()
        print("Controller stopped. Stop commands sent; serial port released.")


if __name__ == "__main__":
    main()
