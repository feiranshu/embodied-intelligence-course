"""Compare nominal URDF FK vs hand_eye.json calibrated FK on measured physical poses."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor.scs215.advanced import Kinematics, load_advanced

adv = load_advanced(ROOT / "vendor" / "scs215" / "advanced.json")
hand_eye = json.loads((ROOT / "config" / "calibration_reference" / "hand_eye.json").read_text(encoding="utf-8"))

nom = Kinematics(adv)
cal = nom.with_joint_calibration(hand_eye["calibrated_joint_mapping"])

poses = {
    "rest_whiteboard": [1010, 898, 60, 679, 393, 275],
    "above_table_142": [993, 903, 142, 699, 291, 282],
    "near_table_222":  [973, 914, 222, 720, 186, 281],
    "touch_table_267": [973, 899, 267, 713, 185, 282],
    "preset_init_300": [950, 920, 300, 740, 80, 276],
    "preset_center":   [950, 896, 319, 713, 97, 272],
}

print(f"{'Pose':<18} | {'Nominal FK [x, y, z] mm':<28} | {'Calibrated FK [x, y, z] mm'}")
print("-" * 82)
for name, raw in poses.items():
    p_nom = [round(float(v), 1) for v in nom.forward(raw)]
    p_cal = [round(float(v), 1) for v in cal.forward(raw)]
    print(f"{name:<18} | {str(p_nom):<28} | {p_cal}")
