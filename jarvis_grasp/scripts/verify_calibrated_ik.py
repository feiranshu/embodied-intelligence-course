"""Read-only verification of calibrated FK/IK directions around our measured tabletop poses."""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor.scs215.advanced import Kinematics, load_advanced

adv = load_advanced(ROOT / "vendor" / "scs215" / "advanced.json")
hand_eye = json.loads((ROOT / "config" / "calibration_reference" / "hand_eye.json").read_text(encoding="utf-8"))
cal = Kinematics(adv).with_joint_calibration(hand_eye["calibrated_joint_mapping"])

observe_raw = np.array([990, 905, 145, 700, 285, 280])
xyz0 = cal.forward(observe_raw)
print("Observe pose raw:", observe_raw.tolist(), "Calibrated XYZ (mm):", np.round(xyz0, 2).tolist())

for name, delta in [
    ("UP +20mm (+Z)", [0, 0, 20]),
    ("DOWN -20mm (-Z)", [0, 0, -20]),
    ("DOWN -80mm (pre-grasp)", [0, 0, -80]),
    ("DOWN -130mm (grasp ~70mm)", [0, 0, -130]),
    ("LEFT +20mm (+Y)", [0, 20, 0]),
    ("RIGHT -20mm (-Y)", [0, -20, 0]),
    ("FWD +20mm (+X)", [20, 0, 0]),
    ("BACK -20mm (-X)", [-20, 0, 0]),
]:
    try:
        sol = cal.inverse(xyz0 + np.array(delta, float), observe_raw, allow_crossing=False)
        xyz1 = cal.forward(sol)
        diff = (sol - observe_raw).tolist()
        print(f"  {name:<26} -> raw={sol.tolist()} delta_raw={diff} xyz={np.round(xyz1, 1).tolist()}")
    except Exception as err:
        print(f"  {name:<26} -> ERROR: {err}")
