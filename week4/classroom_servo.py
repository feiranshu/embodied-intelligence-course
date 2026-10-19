"""SCS215 双舵机课堂实验；仅依赖 pyserial，不使用 lerobot。"""

import argparse
import math
import time

import serial
from serial_port import resolve_port


IDS = (1, 2)
RANGES = (90, 45)
STEPS_PER_DEGREE = 1023 / 300


def packet(servo_id, instruction, params=()):
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([255, 255, *body, (~sum(body)) & 255])


def request(ser, servo_id, instruction, params=(), count=0):
    ser.reset_input_buffer()
    sent = packet(servo_id, instruction, params)
    ser.write(sent)
    deadline = time.monotonic() + 0.25
    data = bytearray()
    while time.monotonic() < deadline:
        data.extend(ser.read(max(1, ser.in_waiting)))
        while len(data) >= 4:
            if data[:2] != b'\xff\xff' or data[3] < 2:
                del data[0]
                continue
            size = data[3] + 4
            if len(data) < size:
                break
            reply = bytes(data[:size])
            del data[:size]
            if reply == sent:
                continue
            if reply[2] != servo_id or sum(reply[2:]) & 255 != 255:
                continue
            if reply[4]:
                raise RuntimeError(f'ID {servo_id} 舵机报错：0x{reply[4]:02X}')
            if reply[3] != count + 2:
                raise RuntimeError(f'ID {servo_id} 应答长度不符：{reply.hex(" ")}')
            return reply[5:-1]
    raise TimeoutError(f'ID {servo_id} 没有有效应答，请检查供电、接线、ID 和波特率')


def read(ser, servo_id, address, count):
    return request(ser, servo_id, 2, [address, count], count)


def write(ser, servo_id, address, values):
    # WRITE 可能按返回级别不应答，因此以 READ 读回结果为准。
    ser.write(packet(servo_id, 3, [address, *values]))
    ser.flush()
    time.sleep(0.05)
    if read(ser, servo_id, address, len(values)) != bytes(values):
        raise RuntimeError(f'ID {servo_id} 地址 0x{address:02X} 写入后读回不一致')


def word(value):
    return list(value.to_bytes(2, 'big'))


def position(ser, servo_id):
    value = int.from_bytes(read(ser, servo_id, 56, 2), 'big')
    if not 0 <= value <= 1023:
        raise RuntimeError(f'位置 {value} 超出 SCS215 范围，请核对型号和字节序')
    return value


def limits(ser, servo_id):
    data = read(ser, servo_id, 9, 4)
    return int.from_bytes(data[:2], 'big'), int.from_bytes(data[2:], 'big')


def scan(ser, first, last):
    found = []
    for servo_id in range(first, last + 1):
        try:
            data = read(ser, servo_id, 3, 4)
            if data[2] != servo_id:
                raise RuntimeError('ID 寄存器不匹配')
            print(f'ID={servo_id}, 型号字节={data[:2].hex(" ")}, 波特率代码={data[3]}', flush=True)
            found.append(servo_id)
        except TimeoutError:
            pass
    print(f'通过校验的舵机 ID：{found}（同 ID 多舵机无法据此区分）')
    return found


def set_id(ser, old_id, new_id):
    if input('请确保总线上只接一个 SCS215 舵机，输入 SINGLE 继续：').strip() != 'SINGLE':
        raise ValueError('未确认单舵机连接，取消编号')
    if read(ser, old_id, 5, 1) != bytes([old_id]):
        raise RuntimeError('原 ID 读回不匹配')
    write(ser, old_id, 40, [0])
    write(ser, old_id, 48, [0])
    current_id = old_id
    try:
        ser.write(packet(old_id, 3, [5, new_id]))
        ser.flush()
        time.sleep(0.1)
        if read(ser, new_id, 5, 1) != bytes([new_id]):
            raise RuntimeError('新 ID 读回不匹配')
        current_id = new_id
    finally:
        write(ser, current_id, 48, [1])
    print(f'已编号并读回确认：{old_id} -> {new_id}')


def setup(ser, zeros):
    widths = [round(degrees * STEPS_PER_DEGREE) for degrees in RANGES]
    if any(z < 0 or z + w > 1023 for z, w in zip(zeros, widths)):
        raise ValueError('零点及最大位置必须在 0～1023 内')
    for servo_id in IDS:
        position(ser, servo_id)
    for servo_id, zero, width in zip(IDS, zeros, widths):
        print(f'ID {servo_id} 原限位={limits(ser, servo_id)}，新限位=({zero}, {zero + width})')
        write(ser, servo_id, 40, [0])
        write(ser, servo_id, 48, [0])
        try:
            write(ser, servo_id, 9, word(zero) + word(zero + width))
        finally:
            write(ser, servo_id, 48, [1])
        print(f'ID {servo_id} 限位读回确认：{limits(ser, servo_id)}')


def targets(values, bounds):
    if len(values) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('必须输入两个 0～1 的有限数值')
    result = []
    for value, (low, high), degrees in zip(values, bounds, RANGES):
        if not 0 <= low < high <= 1023 or high - low != round(degrees * STEPS_PER_DEGREE):
            raise ValueError('舵机限位尚未设为 90°/45°，请先运行 setup')
        result.append(round(low + value * (high - low)))
    return result


def move(ser, values, duration):
    bounds = [limits(ser, i) for i in IDS]
    goal = targets(values, bounds)
    current = [position(ser, i) for i in IDS]
    if any(not low <= p <= high for p, (low, high) in zip(current, bounds)):
        raise ValueError('当前位置不在设定范围内；请先释放扭矩并手动移入范围，或重新选择零点')
    try:
        for i, p in zip(IDS, current):
            write(ser, i, 40, [0])
            write(ser, i, 42, word(p) + word(0) + word(0))
        for i in IDS:
            write(ser, i, 40, [1])
        params = [42, 6]
        for i, p in zip(IDS, goal):
            params.extend([i, *word(p), *word(duration), *word(0)])
        sent = packet(254, 0x83, params)
        ser.write(sent)
        ser.flush()
        print(f'SYNC_WRITE：{sent.hex(" ")}')
        time.sleep(duration / 1000 + 0.5)
        actual = [position(ser, i) for i in IDS]
        for i, value, degrees, target, p in zip(IDS, values, RANGES, goal, actual):
            print(f'ID {i}: 输入={value}, 相对角度={value * degrees:g}°, 目标={target}, 实测={p}, 误差={p - target} 格')
        if any(abs(p - t) > 5 for p, t in zip(actual, goal)):
            raise RuntimeError('至少一个舵机位置误差超过 5 格，请检查运动受阻、供电或时间设置')
        print('两个舵机位置反馈均在目标 ±5 格内；动作后保持位置，退出程序释放扭矩。')
    except BaseException:
        release(ser)
        raise


def release(ser):
    ser.write(packet(254, 0x83, [40, 1, 1, 0, 2, 0]))
    ser.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='auto', help='默认自动选择唯一的 CH343；也可指定 COM 口')
    parser.add_argument('--baud', type=int, default=1000000)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('scan', help='只读扫描，不改变配置和位置')
    p.add_argument('--first', type=int, default=0, choices=range(254), metavar='0..253')
    p.add_argument('--last', type=int, default=6, choices=range(254), metavar='0..253')
    p = sub.add_parser('set-id', help='单独连接一个 SCS215 后编号')
    p.add_argument('old', type=int, choices=range(254), metavar='OLD_ID')
    p.add_argument('new', type=int, choices=range(254), metavar='NEW_ID')
    p = sub.add_parser('setup', help='设置 90°/45°硬件限位，不发运动指令')
    p.add_argument('--zero', type=int, nargs=2, required=True, metavar=('ZERO1', 'ZERO2'))
    sub.add_parser('status', help='读取位置和硬件限位')
    p = sub.add_parser('control', help='交互输入两个 0～1 的比例')
    p.add_argument('--time-ms', type=int, default=2000)
    args = parser.parse_args()
    if args.command == 'control' and not 100 <= args.time_ms <= 30000:
        parser.error('--time-ms 必须在 100～30000 范围内')
    try:
        with serial.Serial(resolve_port(args.port), args.baud, timeout=0.02, write_timeout=1) as ser:
            if args.command == 'scan':
                if not scan(ser, args.first, args.last):
                    raise RuntimeError('未找到通过校验的舵机')
            elif args.command == 'set-id':
                set_id(ser, args.old, args.new)
            elif args.command == 'setup':
                setup(ser, args.zero)
            elif args.command == 'status':
                for i in IDS:
                    print(f'ID {i}: 位置={position(ser, i)}, 限位={limits(ser, i)}')
            else:
                try:
                    while True:
                        line = input('输入两个比例，例如 0.5 0.5；q 退出：').strip()
                        if line.lower() == 'q':
                            break
                        try:
                            values = [float(v) for v in line.replace(',', ' ').split()]
                            if len(values) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                                raise ValueError('必须输入两个 0～1 的有限数值')
                        except ValueError as exc:
                            print(f'输入错误：{exc}')
                            continue
                        move(ser, values, args.time_ms)
                finally:
                    release(ser)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'失败：{exc}\n')
    except (KeyboardInterrupt, EOFError):
        print('\n已退出。')


if __name__ == '__main__':
    main()
