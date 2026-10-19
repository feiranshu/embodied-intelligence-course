"""课堂协议样例：读取 ID 1 的当前位置。"""

import argparse
import serial

from serial_port import resolve_port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="auto", help="默认自动选择唯一的 CH343")
    args = parser.parse_args()
    with serial.Serial(resolve_port(args.port), 1_000_000, timeout=0.2) as bus:
        bus.write(bytes.fromhex("FF FF 01 04 02 38 02 BE"))
        reply = bus.read(8)
    if len(reply) != 8:
        raise RuntimeError(f"舵机应答不完整：{reply.hex(' ')}")
    position = (reply[5] << 8) | reply[6]
    print("当前位置：", position)


if __name__ == "__main__":
    main()
