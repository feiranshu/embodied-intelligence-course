import argparse
import time
import serial

from serial_port import resolve_port

def write_command(servo_id, address, value):
    body = [servo_id, 4, 3, address, value]
    checksum = (~sum(body)) & 0xFF
    return bytes([0xFF, 0xFF, *body, checksum])


def main():
    parser = argparse.ArgumentParser(description="课堂协议样例：单舵机修改 ID（写 EEPROM）")
    parser.add_argument("old_id", type=int)
    parser.add_argument("new_id", type=int)
    parser.add_argument("--port", default="auto", help="默认自动选择唯一的 CH343")
    args = parser.parse_args()
    with serial.Serial(resolve_port(args.port), 1_000_000, timeout=0.2) as ser:
        commands = [
            write_command(args.old_id, 0x28, 0),
            write_command(args.old_id, 0x30, 0),
            write_command(args.old_id, 0x05, args.new_id),
            write_command(args.new_id, 0x30, 1),
        ]
        for command in commands:
            ser.write(command)
            ser.read(6)
            time.sleep(0.1)
    print(f"舵机 ID 已从 {args.old_id} 改为 {args.new_id}")


if __name__ == "__main__":
    main()
