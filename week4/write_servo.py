"""课堂协议样例：将 ID 1 移到原始位置 400。"""

import argparse
import time
import serial

from serial_port import resolve_port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="auto", help="默认自动选择唯一的 CH343")
    args = parser.parse_args()
    with serial.Serial(resolve_port(args.port), 1_000_000, timeout=0.2) as bus:
        try:
            bus.write(bytes.fromhex("FF FF 01 04 03 28 01 CE"))
            bus.read(6)
            bus.write(bytes.fromhex("FF FF 01 05 03 2A 01 90 3B"))
            bus.read(6)
            time.sleep(2)
        finally:
            bus.write(bytes.fromhex("FF FF 01 04 03 28 00 CF"))


if __name__ == "__main__":
    main()
