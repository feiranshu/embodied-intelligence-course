"""Measure exact HSV statistics of the real green pencil pouch vs wood tabletop in hello_world_observe.png."""

import cv2
import numpy as np
from pathlib import Path

img_path = Path(__file__).resolve().parents[1] / "logs" / "hello_world_observe.png"
img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Pouch patch (top-left fabric, avoiding zipper)
pouch = hsv[15:85, 25:125]
# Wood tabletop patch (center)
wood = hsv[200:320, 220:420]
# White gripper jaw patch (top-center)
jaw = hsv[5:45, 270:350]

for name, patch in [("pouch_green", pouch), ("wood_table", wood), ("white_jaw", jaw)]:
    h, s, v = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    print(
        f"{name:<12}: H=[p5={np.percentile(h,5):.1f}, med={np.median(h):.1f}, p95={np.percentile(h,95):.1f}] "
        f"S=[p5={np.percentile(s,5):.1f}, med={np.median(s):.1f}, p95={np.percentile(s,95):.1f}] "
        f"V=[p5={np.percentile(v,5):.1f}, med={np.median(v):.1f}, p95={np.percentile(v,95):.1f}]"
    )

# Test clean segmentation of sage/teal/green objects excluding top gripper zone & wood
mask = cv2.inRange(hsv, (40, 38, 45), (105, 255, 235))
# Mask out gripper white jaws (low saturation already excluded, also ignore y < 65 around x in [210..520] only if bright)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(f"Found {len(contours)} green-teal contours:")
vis = img.copy()
for idx, cnt in enumerate(contours[:5]):
    area = cv2.contourArea(cnt)
    if area < 400:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    box = np.int32(cv2.boxPoints(rect))
    cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 180, 0), 2)
    cx, cy = int(rect[0][0]), int(rect[0][1])
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
    print(f"  #{idx+1}: area={area:.0f} bbox=({x},{y},{w},{h}) center=({cx},{cy}) minAreaRect=({rect[1][0]:.1f}x{rect[1][1]:.1f}, angle={rect[2]:.1f})")

out = img_path.with_name("hello_world_pouch_seg.png")
ok, buf = cv2.imencode(".png", vis)
out.write_bytes(buf.tobytes())
print("Saved segmentation check to", out)
