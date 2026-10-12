import time
import serial

PORT = "/dev/cu.usbmodemXXXX"
OLD_ID = 1
NEW_ID = 2

def write_command(servo_id, address, value):
    body = [servo_id, 4, 3, address, value]
    checksum = (~sum(body)) & 0xFF
    return bytes([0xFF, 0xFF, *body, checksum])


with serial.Serial(PORT, 1_000_000, timeout=0.2) as ser:
    commands = [
        write_command(OLD_ID, 0x28, 0),  # 关闭扭矩
        write_command(OLD_ID, 0x30, 0),  # 解锁 EEPROM
        write_command(OLD_ID, 0x05, NEW_ID),  # 写入新 ID
        write_command(NEW_ID, 0x30, 1),  # 使用新 ID 锁定 EEPROM
    ]

    for command in commands:
        ser.write(command)
        ser.read(6)
        time.sleep(0.1)

print(f"舵机 ID 已从 {OLD_ID} 改为 {NEW_ID}")