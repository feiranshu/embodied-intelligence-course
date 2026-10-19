# Maintenance instructions

Read `README.md` and `HANDOFF.md` before changing or running this project.

- Hardware is six Feetech SCS215 servos (observed model 8965), not STS3215.
- Do not run motion commands or write EEPROM unless the user explicitly requests it,
  is beside the arm, and can cut external servo power.
- Default to offline tests and `--dry-run`; `status` is the read-only hardware check.
- Preserve torque cleanup on success, exception, and keyboard interruption.
- Calibration intervals used by linear position control must not cross 0/1023.
- After Python edits, run `python -m py_compile homework_control.py servo_bus.py`
  and `python -m unittest -v test_offline.py` without hardware.
- After configuration edits, run `python -m json.tool config.json`.

