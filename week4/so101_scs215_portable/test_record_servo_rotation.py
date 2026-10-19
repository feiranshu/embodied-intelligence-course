"""Offline checks: raw-data retention and release/focus/fault stopping; no hardware."""
import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from record_servo_rotation import (Journal, PositionRecorder, KeyboardControl,
                                   LoggedSerial, restore_limits, main)


def status_packet(raw, status=0):
    body = [5,4,status,raw >> 8, raw & 255]
    return bytes([255,255,*body,(~sum(body)) & 255])


class RecorderTests(unittest.TestCase):
    def test_keeps_duplicates_endpoints_jumps_out_of_range_and_failed_replies(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            journal = Journal(Path(folder)/'capture')
            serial = Mock()
            serial.in_waiting = 0
            broken = bytearray(status_packet(43)); broken[-1] ^= 1
            serial.read.side_effect = [status_packet(n) for n in [1022,1022,1023,0,0,700,1300]] + [bytes(broken),b'']
            bus = Mock(lock=threading.RLock(),ser=LoggedSerial(serial,journal))
            recorder = PositionRecorder(bus,5,150,50,journal)
            for _ in range(9):
                recorder.sample()
            recorder.close(); journal.file.close()
            with (journal.folder/'positions.csv').open(encoding='utf-8-sig',newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r['raw_position'] for r in rows],['1022','1022','1023','0','0','700','1300','',''])
            self.assertEqual(rows[7]['raw_candidate'],'43')
            self.assertEqual(rows[6]['within_0_1023'],'False')
            self.assertTrue(rows[7]['error']); self.assertTrue(rows[8]['error'])
            self.assertEqual(serial.write.call_count,9)  # Failed reads not silently retried.
            events = [json.loads(s) for s in (journal.folder/'serial_and_events.jsonl').read_text().splitlines()]
            self.assertEqual(len([e for e in events if e['kind']=='serial_rx']),9)

    def test_focus_loss_and_fault_require_release_and_good_read_before_restart(self):
        recorder = Mock(joint=5,speed=150,direction='stopped',failure=None,latest={'error':''})
        recorder.fault = threading.Event()
        bus = Mock(); bus.stop.return_value = []
        control = KeyboardControl(bus,recorder)
        released = dict(focused=True,left=False,right=False,stop=False,quit=False)
        held = dict(released,left=True)
        control.tick(released); control.tick(held)
        self.assertEqual(bus.run.call_count,1)
        control.tick(dict(held,focused=False))
        bus.stop.assert_called_with(5)
        control.tick(held)
        self.assertEqual(bus.run.call_count,1)
        control.tick(released); control.tick(held)
        self.assertEqual(bus.run.call_count,2)
        recorder.latest = {'error':'timeout'}; recorder.fault.set()
        control.tick(held); control.tick(released); control.tick(held)
        self.assertEqual(bus.run.call_count,2)
        recorder.latest = {'error':''}
        control.tick(held)  # Successful read alone must not restart a held key.
        self.assertEqual(bus.run.call_count,2)
        control.tick(released); control.tick(released); control.tick(held)
        self.assertEqual(bus.run.call_count,3)

    def test_stop_failure_is_reported(self):
        recorder = Mock(joint=5,direction='left')
        bus = Mock(); bus.stop.return_value = ['torque readback failed']
        with self.assertRaisesRegex(RuntimeError,'NOT confirmed'):
            KeyboardControl(bus,recorder).stop()

    def test_restores_exact_original_limits_instead_of_default_position_limits(self):
        bus = Mock(lock=threading.RLock())
        bus.mode.return_value = ('position',42,900)
        restore_limits(bus,5,(42,900))
        writes = bus._write.call_args_list
        self.assertEqual([call.args for call in writes],
                         [(5,48,[0]),(5,9,[0,42]),(5,11,[3,132]),(5,48,[1])])

    def test_record_only_session_never_sends_motion_or_mode_commands(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            bus = Mock(lock=threading.RLock(),port='FAKE')
            bus.ser.in_waiting = 0
            bus.ser.read.return_value = status_packet(1022)
            bus.mode.return_value = ('wheel',0,0)
            output = Path(folder)/'session'
            with patch('record_servo_rotation.ServoBus',return_value=bus):
                code = main(['--id','5','--record-only','--duration','.08','--output',str(output)])
            self.assertEqual(code,0)
            bus.run.assert_not_called(); bus.set_mode.assert_not_called(); bus.stop.assert_not_called()
            data = json.loads((output/'session.json').read_text())
            self.assertGreater(data['samples'],0)
            self.assertTrue(data['read_only'])


if __name__ == '__main__':
    unittest.main()
