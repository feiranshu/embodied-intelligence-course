"""Resolve the classroom examples' serial port without a machine-specific default."""

from serial.tools import list_ports


def resolve_port(requested="auto"):
    if requested and requested.lower() != "auto":
        return requested
    ports = list(list_ports.comports())
    matches = [port.device for port in ports if (port.vid, port.pid) == (0x1A86, 0x55D3)]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(port.device for port in ports) or "无"
    if matches:
        raise ValueError(f"检测到多个 CH343 串口：{', '.join(matches)}；请用 --port 指定")
    raise ValueError(f"未找到 CH343 转接板（可用串口：{available}）；请检查连接或用 --port 指定")
