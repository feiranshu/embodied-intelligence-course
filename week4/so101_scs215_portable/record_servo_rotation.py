"""Record every polled SCS215 position, with keyboard wheel control on Windows."""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

from servo_bus import (ServoBus, load_config, make_packet, PRESENT_POSITION,
                       LOCK, MIN_POSITION_LIMIT, MAX_POSITION_LIMIT)

ROOT = Path(__file__).resolve().parent


class Journal:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.start = time.perf_counter()
        self.lock = threading.Lock()
        self.file = (self.folder / 'serial_and_events.jsonl').open('w', encoding='utf-8')

    def event(self, kind, **fields):
        with self.lock:
            row = dict(utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_s=time.perf_counter()-self.start, kind=kind, **fields)
            self.file.write(json.dumps(row, ensure_ascii=False)+'\n')
            self.file.flush()


class LoggedSerial:
    """Keep all actual wire reads/writes, including retries made by existing APIs."""
    def __init__(self, serial_port, journal):
        self.wrapped, self.journal = serial_port, journal

    def __getattr__(self, key):
        return getattr(self.wrapped, key)

    def write(self, data):
        self.journal.event('serial_tx', hex=bytes(data).hex(' '))
        try:
            return self.wrapped.write(data)
        except Exception as error:
            self.journal.event('serial_write_error', error=str(error))
            raise

    def read(self, count):
        started = time.perf_counter()-self.journal.start
        try:
            reply = self.wrapped.read(count)
        except Exception as error:
            self.journal.event('serial_read_error', read_started_s=started, error=str(error))
            raise
        self.journal.event('serial_rx', requested_bytes=count,
                           read_started_s=started, hex=bytes(reply).hex(' '))
        return reply

    def reset_input_buffer(self):
        # Preserve late/unsolicited bytes that the normal protocol would discard.
        pending = self.wrapped.in_waiting
        if pending:
            data = self.wrapped.read(pending)
            self.journal.event('serial_pending', hex=data.hex(' '))
        # Do not reset after draining: bytes arriving later must not be silently erased.


def decode_position(reply, joint):
    """Decode without clipping, wrapping, rejecting jumps, or hiding fault bytes."""
    raw_candidate = (reply[5] << 8) | reply[6] if len(reply) == 8 else ''
    status = reply[4] if len(reply) >= 5 else ''
    error = ''
    if len(reply) != 8:
        error = f'expected 8 bytes, got {len(reply)}'
    elif reply[:2] != b'\xff\xff' or reply[2] != joint or reply[3] != 4:
        error = 'invalid header, servo ID, or length'
    elif sum(reply[2:]) & 255 != 255:
        error = 'checksum mismatch'
    elif status:
        error = f'servo status error 0x{status:02x}'
    raw = raw_candidate if not error else ''
    return raw, raw_candidate, status, error


class PositionRecorder:
    FIELDS = ['sample', 'utc', 'request_started_s', 'reply_finished_s', 'read_ms',
              'joint', 'command_direction', 'speed_parameter', 'raw_position',
              'raw_candidate', 'within_0_1023', 'status_byte', 'reply_hex', 'error']

    def __init__(self, bus, joint, speed, hz, journal):
        self.bus, self.joint, self.speed = bus, joint, speed
        self.period, self.journal = 1/hz, journal
        self.stop_event = threading.Event()
        self.fault = threading.Event()
        self.direction = 'stopped'
        self.count = self.errors = 0
        self.latest = None
        self.failure = None
        self.file = (journal.folder/'positions.csv').open('w', encoding='utf-8-sig', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.file.flush()
        self.thread = threading.Thread(target=self.run, name='raw-position-recorder', daemon=True)

    def sample(self):
        reply = b''
        raw = candidate = status = ''
        # Timestamp actual transaction after acquiring the shared serial lock.
        with self.bus.lock:
            direction = self.direction
            started = time.perf_counter()-self.journal.start
            utc = datetime.now(timezone.utc).isoformat()
            try:
                self.bus.ser.reset_input_buffer()
                self.bus.ser.write(make_packet(self.joint, 2, [PRESENT_POSITION, 2]))
                self.bus.ser.flush()
                reply = self.bus.ser.read(8)  # Exactly one attempt; no hidden position retries.
                raw, candidate, status, error = decode_position(reply, self.joint)
            except Exception as exc:
                error = str(exc)
            ended = time.perf_counter()-self.journal.start
        self.count += 1
        row = dict(sample=self.count, utc=utc, request_started_s=started,
                   reply_finished_s=ended, read_ms=(ended-started)*1000,
                   joint=self.joint, command_direction=direction,
                   speed_parameter=self.speed if direction != 'stopped' else 0,
                   raw_position=raw, raw_candidate=candidate,
                   within_0_1023=(0 <= raw <= 1023) if raw != '' else '',
                   status_byte=status, reply_hex=reply.hex(' '), error=error)
        self.writer.writerow(row)
        self.file.flush()  # Repeated/endpoint/error rows are all written immediately.
        self.latest = row
        if error:
            self.errors += 1
            self.fault.set()

    def run(self):
        try:
            while not self.stop_event.is_set():
                started = time.perf_counter()
                self.sample()
                self.stop_event.wait(max(0, self.period-(time.perf_counter()-started)))
        except Exception as error:
            self.failure = error
            self.fault.set()

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.file.close()


class WindowsKeys:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('Keyboard control requires Windows; use --record-only elsewhere.')
        from ctypes import wintypes
        self.api = ctypes.WinDLL('user32', use_last_error=True)
        self.api.GetForegroundWindow.restype = wintypes.HWND
        self.api.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.api.GetAsyncKeyState.restype = ctypes.c_short
        self.window = self.api.GetForegroundWindow()
        if not self.window:
            raise RuntimeError('No foreground terminal window detected.')

    def read(self):
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()  # Consume console input, including Windows arrow prefixes.
        focused = self.api.GetForegroundWindow() == self.window
        down = lambda key: bool(self.api.GetAsyncKeyState(key) & 0x8000)
        return dict(focused=focused, left=down(0x25), right=down(0x27),
                    stop=down(0x20), quit=down(0x1b) or down(ord('Q')), mark=down(ord('M')))


class KeyboardControl:
    def __init__(self, bus, recorder):
        self.bus, self.recorder = bus, recorder
        self.active = None
        self.armed = False

    def stop(self):
        failures = self.bus.stop(self.recorder.joint)
        self.recorder.direction = 'stopped'
        self.active = None
        self.recorder.journal.event('stop', failures=failures)
        if failures:
            raise RuntimeError('Stop NOT confirmed: '+'; '.join(failures))

    def tick(self, keys):
        released = not keys['left'] and not keys['right']
        if self.recorder.fault.is_set():
            self.armed = False
            if self.active is not None:
                self.stop()
            if self.recorder.failure:
                raise RuntimeError(f'Recorder stopped: {self.recorder.failure}')
            if released and self.recorder.latest and not self.recorder.latest['error']:
                self.recorder.fault.clear()
            return
        inhibited = not keys['focused'] or keys['stop'] or keys['quit']
        if inhibited:
            self.armed = False
        elif released and self.recorder.latest and not self.recorder.latest['error']:
            self.armed = True
        desired = None
        if self.armed and not inhibited and keys['left'] != keys['right']:
            desired = 'left' if keys['left'] else 'right'
        if desired != self.active:
            if self.active is not None:
                self.stop()
            if desired:
                self.bus.run(self.recorder.joint, desired, self.recorder.speed, hold_others=True)
                self.recorder.direction = desired
                self.active = desired
                self.recorder.journal.event('motion', direction=desired, speed=self.recorder.speed)
        if self.active:
            self.bus.heartbeat(self.recorder.joint)


def restore_limits(bus, joint, limits):
    with bus.lock:
        bus._write(joint, LOCK, [0])
        try:
            for address, value in zip((MIN_POSITION_LIMIT, MAX_POSITION_LIMIT), limits):
                bus._write(joint, address, [value >> 8, value & 255])
        finally:
            bus._write(joint, LOCK, [1])
        if bus.mode(joint)[1:] != tuple(limits):
            raise RuntimeError('Original mode/limits restoration was not confirmed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', type=int, choices=range(1,7), required=True, help='selected servo ID')
    parser.add_argument('--port', help='auto detect by default; e.g. COM8')
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--speed', type=int, default=150, help='wheel speed parameter, 100..1000')
    parser.add_argument('--hz', type=float, default=50, help='requested poll rate; actual rate is logged')
    parser.add_argument('--timeout', type=float, default=.1, help='serial read timeout in seconds')
    parser.add_argument('--duration', type=float, default=0, help='optional duration, seconds; 0 until Q/Ctrl+C')
    parser.add_argument('--output', type=Path, help='new, not yet existing output folder')
    parser.add_argument('--record-only', action='store_true', help='read only, no mode/torque/motion writes')
    parser.add_argument('--set-wheel-mode', action='store_true',
                        help='allow selected servo mode change; restore exact original limits on exit')
    args = parser.parse_args(argv)
    if (not 100 <= args.speed <= 1000 or not 0 < args.hz <= 200
            or not 0 < args.timeout <= .5 or args.duration < 0):
        parser.error('speed:100..1000, hz:(0,200], timeout:(0,0.5], duration:>=0')
    if args.record_only and args.set_wheel_mode:
        parser.error('--record-only cannot be combined with --set-wheel-mode')
    keys = None if args.record_only else WindowsKeys()
    config = load_config(args.config)
    config['servos'] = [item for item in config['servos'] if int(item['id']) == args.id]
    if not config['servos']:
        parser.error('selected ID is missing from config')
    folder = args.output or ROOT/'rotation_records'/f"id{args.id}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    journal = Journal(folder)
    bus = recorder = control = None
    original_limits = None
    restore_needed = False
    failures = []
    result = dict(id=args.id, requested_hz=args.hz, speed_parameter=args.speed,
                  read_only=args.record_only, started_utc=datetime.now(timezone.utc).isoformat())
    print(f'Output: {folder.resolve()}', flush=True)
    if args.record_only:
        print('只读记录模式：Ctrl+C退出，不发送运动、扭矩或模式设置指令。', flush=True)
    else:
        print('按住左/右方向键转动，松键/空格/切换窗口停止；M=标记，Q/Esc=退出。', flush=True)
        print('通过实物判断是否完成一圈；程序不会将反馈回到起点认定为转满360度。', flush=True)
        print('仅控制选定ID，其他关节保持原状态；请托稳机械臂，观察连杆和线缆。', flush=True)
    try:
        bus = ServoBus(config, args.port, stop_on_close=False)
        bus.ser.timeout = args.timeout
        bus.ser = LoggedSerial(bus.ser, journal)
        result['port'] = bus.port
        mode, low, high = bus.mode(args.id)
        original_limits = (low,high)
        result['original_mode_limits'] = [mode,low,high]
        # Persist recovery information before any mode write, including interrupted startup.
        (folder/'session.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        if not args.record_only:
            failures_stop = bus.stop(args.id)
            if failures_stop:
                raise RuntimeError('; '.join(failures_stop))
            if mode != 'wheel':
                if not args.set_wheel_mode:
                    raise RuntimeError('Selected servo is not in wheel mode. Add --set-wheel-mode to allow switching.')
                restore_needed = True
                bus.set_mode(args.id, 'wheel', hold_others=True)
        recorder = PositionRecorder(bus,args.id,args.speed,args.hz,journal)
        recorder.thread.start()
        control = None if args.record_only else KeyboardControl(bus,recorder)
        mark_down = False
        shown = 0.
        while not args.duration or time.perf_counter()-journal.start < args.duration:
            if keys:
                state = keys.read()
                control.tick(state)
                if state['quit'] and state['focused']:
                    break
                marked = state['mark'] and state['focused']
                if marked and not mark_down:
                    journal.event('operator_mark', latest_sample=recorder.count)
                    print('\nMARK saved', flush=True)
                mark_down = marked
            elif recorder.failure:
                raise RuntimeError(f'Recorder stopped: {recorder.failure}')
            if time.perf_counter()-shown >= .25 and recorder.latest:
                row = recorder.latest
                print(f"\rID{args.id} samples={recorder.count} errors={recorder.errors} raw={str(row['raw_position']):>5} "
                      f"direction={recorder.direction:7} error={row['error'][:65]:65}",end='',flush=True)
                shown = time.perf_counter()
            time.sleep(.01)
    except KeyboardInterrupt:
        journal.event('keyboard_interrupt')
    except Exception as error:
        failures.append(str(error))
        journal.event('session_error',error=str(error))
    finally:
        # Keep recording through the final stop; release selected motor only.
        if bus and not args.record_only:
            try:
                if control:
                    control.stop()
                else:
                    stop_errors = bus.stop(args.id)
                    if stop_errors:
                        raise RuntimeError('; '.join(stop_errors))
                result['stop_confirmed'] = True
            except Exception as error:
                result['stop_confirmed'] = False
                failures.append(f'STOP NOT CONFIRMED: {error}')
        if recorder:
            recorder.close()
            result.update(samples=recorder.count, read_errors=recorder.errors)
            if recorder.failure:
                failures.append(f'recording failure: {recorder.failure}')
        if bus and restore_needed and result.get('stop_confirmed'):
            try:
                restore_limits(bus,args.id,original_limits)
                result['original_mode_restored'] = True
            except Exception as error:
                result['original_mode_restored'] = False
                failures.append(f'MODE RESTORATION FAILED: {error}')
        elif restore_needed:
            result['original_mode_restored'] = False
            failures.append('Mode restoration skipped because stop was not confirmed; original limits are in session.json')
        if bus:
            try:
                bus.close()
            except Exception as error:
                failures.append(f'Serial close failed: {error}')
        elapsed = time.perf_counter()-journal.start
        result.update(elapsed_s=elapsed, errors=failures,
                      average_samples_per_session_second=recorder.count/elapsed if recorder else 0)
        journal.event('session_end', summary=result)
        journal.file.close()
        (folder/'session.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('\nSaved:', folder.resolve())
        for error in failures:
            print('ERROR:',error)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
