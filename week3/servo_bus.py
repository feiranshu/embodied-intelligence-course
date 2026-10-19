"""Minimal Feetech SCS215 protocol support used by this project.

The bus reports model 8965 and uses the SCS protocol with big-endian
16-bit register values.  This module intentionally has no LeRobot or
Feetech SDK dependency.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import serial
from serial.tools import list_ports


MIN_POSITION_LIMIT = 0x09
MAX_POSITION_LIMIT = 0x0B
SERVO_ID = 0x05
TORQUE_ENABLE = 0x28
GOAL_POSITION = 0x2A
RUNNING_TIME = 0x2C
LOCK = 0x30
PRESENT_POSITION = 0x38
POSITION_MODE_MAX = 1023
BROADCAST_ID = 0xFE
SYNC_WRITE = 0x83
WATCHDOG_TIMEOUT_SECONDS = 0.8


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = {"port", "baud_rate", "servos"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"配置缺少字段：{', '.join(sorted(missing))}")
    ids = [int(item["id"]) for item in config["servos"]]
    if len(ids) != len(set(ids)):
        raise ValueError("配置中存在重复舵机 ID")
    return config


def available_ports() -> list[dict]:
    result = []
    for item in list_ports.comports():
        result.append(
            {
                "device": item.device,
                "description": item.description or "",
                "hwid": item.hwid or "",
                "vid": item.vid,
                "pid": item.pid,
            }
        )
    return result


def resolve_port(requested: str) -> tuple[str, list[dict]]:
    ports = available_ports()
    if requested and requested.lower() != "auto":
        return requested, ports

    # Waveshare Bus Servo Adapter (A) used in this project exposes CH343.
    exact = [
        item
        for item in ports
        if item["vid"] == 0x1A86 and item["pid"] == 0x55D3
    ]
    if len(exact) == 1:
        return exact[0]["device"], ports

    keywords = ("ch343", "usb-enhanced-serial", "waveshare")
    likely = [
        item
        for item in ports
        if any(word in (item["description"] + " " + item["hwid"]).lower() for word in keywords)
    ]
    if len(likely) == 1:
        return likely[0]["device"], ports

    visible = "；".join(
        f"{item['device']} ({item['description'] or item['hwid']})" for item in ports
    ) or "未发现串口"
    if not likely:
        raise RuntimeError(
            "无法自动识别 Waveshare/CH343 串口。"
            f"当前串口：{visible}。请在 config.json 中填写明确端口。"
        )
    raise RuntimeError(
        "检测到多个可能的控制板串口，无法自动选择："
        f"{visible}。请在 config.json 中填写明确端口。"
    )


def make_packet(servo_id: int, instruction: int, parameters=()) -> bytes:
    length = len(parameters) + 2
    body = [int(servo_id), length, int(instruction), *parameters]
    checksum = (~sum(body)) & 0xFF
    return bytes([0xFF, 0xFF, *body, checksum])


def _read_status(ser: serial.Serial, expected_length: int) -> bytes:
    reply = ser.read(expected_length)
    if len(reply) != expected_length:
        text = reply.hex(" ") if reply else "无回复"
        raise RuntimeError(f"状态包长度错误：{text}")
    if reply[:2] != b"\xff\xff":
        raise RuntimeError(f"状态包帧头错误：{reply.hex(' ')}")
    if ((sum(reply[2:-1]) + reply[-1]) & 0xFF) != 0xFF:
        raise RuntimeError(f"状态包校验失败：{reply.hex(' ')}")
    if reply[4] != 0:
        raise RuntimeError(f"舵机 ID {reply[2]} 返回错误码 0x{reply[4]:02X}")
    return reply


def request(ser: serial.Serial, packet: bytes, reply_length: int, attempts: int = 3) -> bytes:
    last_error = None
    for attempt in range(attempts):
        ser.reset_input_buffer()
        ser.write(packet)
        ser.flush()
        try:
            return _read_status(ser, reply_length)
        except RuntimeError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.08)
    raise RuntimeError(f"连续 {attempts} 次通信失败：{last_error}")


class ServoBus:
    def __init__(
        self,
        config: dict,
        port_override: str | None = None,
        *,
        stop_on_close: bool = True,
    ):
        self.config = config
        self.items = {int(item["id"]): item for item in config["servos"]}
        self.lock = threading.RLock()
        self.running_id: int | None = None
        self.last_heartbeat = 0.0
        self.stop_on_close = stop_on_close
        self.port, self.detected_ports = resolve_port(port_override or config["port"])
        try:
            self.ser = serial.Serial(
                self.port,
                int(config["baud_rate"]),
                timeout=0.5,
                write_timeout=0.5,
            )
        except serial.SerialException as error:
            raise RuntimeError(
                f"无法打开串口 {self.port}：{error}。"
                "请关闭其他控制器、串口监视器和占用该端口的 Python 程序。"
            ) from error
        time.sleep(0.1)
        self._watchdog_exit = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="servo-motion-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_exit.wait(0.1):
            with self.lock:
                if (
                    self.running_id is not None
                    and time.monotonic() - self.last_heartbeat > WATCHDOG_TIMEOUT_SECONDS
                ):
                    expired_id = self.running_id
                    failures = self._stop_one_locked(expired_id)
                    detail = "；".join(failures) if failures else "已停止"
                    print(f"通信看门狗：ID {expired_id} 心跳超时，{detail}")

    def _item(self, servo_id) -> dict:
        try:
            return self.items[int(servo_id)]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"无效舵机 ID：{servo_id}") from error

    def _write(self, servo_id: int, address: int, values) -> bytes:
        return request(
            self.ser,
            make_packet(servo_id, 0x03, [address, *values]),
            6,
        )

    def _write_without_reply(self, packet: bytes) -> None:
        """Write a broadcast packet, for which the SCS bus sends no status reply."""
        self.ser.reset_input_buffer()
        written = self.ser.write(packet)
        self.ser.flush()
        if written != len(packet):
            raise RuntimeError(f"串口只写入了 {written}/{len(packet)} 字节")

    def _read_u16(self, servo_id: int, address: int) -> int:
        reply = request(
            self.ser,
            make_packet(servo_id, 0x02, [address, 0x02]),
            8,
        )
        return (reply[5] << 8) | reply[6]

    def read_position(self, servo_id: int) -> int:
        return self._read_u16(servo_id, PRESENT_POSITION)

    def sync_write_positions(self, positions: dict[int, int]) -> None:
        """Send one SCS sync-write packet containing every requested goal position.

        The tested SCS215 bus used by this project encodes 16-bit register values
        in big-endian order.  A broadcast sync-write does not return a status packet.
        """
        if not positions:
            raise ValueError("同步写入的位置不能为空")
        parameters = [GOAL_POSITION, 0x02]
        seen = set()
        for raw_id, raw_position in positions.items():
            servo_id = int(raw_id)
            position = int(raw_position)
            if not 1 <= servo_id < BROADCAST_ID:
                raise ValueError(f"无效舅机 ID：{servo_id}")
            if servo_id in seen:
                raise ValueError(f"重复舅机 ID：{servo_id}")
            if not 0 <= position <= POSITION_MODE_MAX:
                raise ValueError(
                    f"ID {servo_id} 目标位置 {position} 超出 0～{POSITION_MODE_MAX}"
                )
            seen.add(servo_id)
            parameters.extend(
                [servo_id, (position >> 8) & 0xFF, position & 0xFF]
            )
        with self.lock:
            self._write_without_reply(make_packet(BROADCAST_ID, SYNC_WRITE, parameters))

    def set_servo_id(self, old_id: int, new_id: int) -> None:
        """Change one attached SCS215 ID and verify communication at the new ID."""
        old_id = int(old_id)
        new_id = int(new_id)
        if not 1 <= old_id < BROADCAST_ID or not 1 <= new_id < BROADCAST_ID:
            raise ValueError("舅机 ID 必须位于 1～253")
        if old_id == new_id:
            raise ValueError("新旧 ID 相同，无需修改")
        with self.lock:
            # ID is an EEPROM field.  Only one motor should be connected here.
            self.set_torque(old_id, False)
            self._write(old_id, LOCK, [0])
            time.sleep(0.1)
            self._write(old_id, SERVO_ID, [new_id])
            time.sleep(0.1)
            self._write(new_id, LOCK, [1])
            self.read_position(new_id)

    def set_torque(self, servo_id: int, enabled: bool) -> None:
        self._write(servo_id, TORQUE_ENABLE, [1 if enabled else 0])

    def mode(self, servo_id: int) -> tuple[str, int, int]:
        minimum = self._read_u16(servo_id, MIN_POSITION_LIMIT)
        maximum = self._read_u16(servo_id, MAX_POSITION_LIMIT)
        mode = "wheel" if minimum == 0 and maximum == 0 else "position"
        return mode, minimum, maximum

    def state(self) -> list[dict]:
        result = []
        with self.lock:
            for item in self.config["servos"]:
                servo_id = int(item["id"])
                mode, minimum, maximum = self.mode(servo_id)
                result.append(
                    {
                        "id": servo_id,
                        "name": item["name"],
                        "label": item.get("label", item["name"]),
                        "position": self.read_position(servo_id),
                        "mode": mode,
                        "min_limit": minimum,
                        "max_limit": maximum,
                        "running": self.running_id == servo_id,
                    }
                )
        return result

    def set_mode(self, selector, mode: str) -> dict:
        if mode not in {"wheel", "position"}:
            raise ValueError("模式必须是 wheel 或 position")
        ids = list(self.items) if selector == "all" else [self._item(selector)["id"]]
        maximum = 0 if mode == "wheel" else POSITION_MODE_MAX
        changed = []
        with self.lock:
            self._stop_all_locked()
            for servo_id in ids:
                servo_id = int(servo_id)
                self.set_torque(servo_id, False)
                self._write(servo_id, LOCK, [0])
                try:
                    self._write(servo_id, MIN_POSITION_LIMIT, [0x00, 0x00])
                    self._write(
                        servo_id,
                        MAX_POSITION_LIMIT,
                        [(maximum >> 8) & 0xFF, maximum & 0xFF],
                    )
                finally:
                    self._write(servo_id, LOCK, [1])
                actual, minimum_read, maximum_read = self.mode(servo_id)
                if actual != mode:
                    raise RuntimeError(
                        f"ID {servo_id} 模式校验失败："
                        f"min={minimum_read}, max={maximum_read}"
                    )
                changed.append(servo_id)
        return {"mode": mode, "ids": changed}

    def run(self, selector, direction: str, speed: int) -> dict:
        item = self._item(selector)
        servo_id = int(item["id"])
        if direction not in {"left", "right"}:
            raise ValueError("方向必须是 left 或 right")
        speed = int(speed)
        if not 100 <= speed <= 1000:
            raise ValueError("连续模式速度必须位于 100～1000")

        with self.lock:
            mode, minimum, maximum = self.mode(servo_id)
            if mode != "wheel":
                raise ValueError(
                    f"ID {servo_id} 尚未切换到连续模式："
                    f"min={minimum}, max={maximum}"
                )
            self._stop_all_locked()
            self.set_torque(servo_id, True)
            physical_direction = direction
            if bool(item.get("invert_direction", False)):
                physical_direction = "right" if direction == "left" else "left"
            value = speed if physical_direction == "left" else speed + 1024
            self._write(
                servo_id,
                RUNNING_TIME,
                [(value >> 8) & 0xFF, value & 0xFF],
            )
            self.running_id = servo_id
            self.last_heartbeat = time.monotonic()
        return {
            "id": servo_id,
            "name": item["name"],
            "direction": direction,
            "speed": speed,
        }

    def heartbeat(self, selector) -> dict:
        servo_id = int(self._item(selector)["id"])
        with self.lock:
            if self.running_id != servo_id:
                raise ValueError(f"ID {servo_id} 当前未运行")
            self.last_heartbeat = time.monotonic()
        return {"id": servo_id}

    def _stop_one_locked(self, servo_id: int) -> list[str]:
        failures = []
        try:
            self._write(servo_id, RUNNING_TIME, [0x00, 0x00])
        except Exception as error:  # Best effort emergency stop.
            failures.append(f"ID {servo_id} 速度归零失败：{error}")
        try:
            self.set_torque(servo_id, False)
        except Exception as error:
            failures.append(f"ID {servo_id} 关闭扭矩失败：{error}")
        if self.running_id == servo_id:
            self.running_id = None
            self.last_heartbeat = 0.0
        return failures

    def _stop_all_locked(self) -> list[str]:
        failures = []
        for servo_id in self.items:
            failures.extend(self._stop_one_locked(servo_id))
        self.running_id = None
        return failures

    def stop(self, selector="all") -> list[str]:
        with self.lock:
            if selector == "all":
                return self._stop_all_locked()
            return self._stop_one_locked(int(self._item(selector)["id"]))

    def close(self) -> None:
        if hasattr(self, "_watchdog_exit"):
            self._watchdog_exit.set()
        try:
            if (
                self.stop_on_close
                and getattr(self, "ser", None)
                and self.ser.is_open
            ):
                self.stop("all")
        finally:
            if getattr(self, "ser", None) and self.ser.is_open:
                self.ser.close()
            if hasattr(self, "_watchdog_thread"):
                self._watchdog_thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
