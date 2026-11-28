"""Extract start, mid, and end keyframes from an MP4 video (supports Chinese Windows paths)."""

import sys
from pathlib import Path
import os
import shutil
import cv2

mp4_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

tmp = Path(os.environ.get("TEMP", ".")) / f"extract_{mp4_path.name}"
shutil.copy2(mp4_path, tmp)
try:
    cap = cv2.VideoCapture(str(tmp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total_frames={total}")
    indices = sorted(set([0, total // 4, total // 2, (3 * total) // 4, max(0, total - 2)]))
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            ok2, buf = cv2.imencode(".png", frame)
            if ok2:
                p = out_dir / f"frame_{idx:04d}.png"
                p.write_bytes(buf.tobytes())
                print("saved", p)
    cap.release()
finally:
    tmp.unlink(missing_ok=True)
