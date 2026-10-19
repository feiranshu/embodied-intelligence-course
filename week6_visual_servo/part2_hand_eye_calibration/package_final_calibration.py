"""Package the selected candidate and preserve both original batches."""
from pathlib import Path
import json
import shutil
import hashlib
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
source=HERE/'final_calibration_20261016_previous'
target=HERE/'最终标定结果'
result=json.loads((source/'hand_eye.json').read_text(encoding='utf-8'))
result['calibration_dataset']={
    'label':'上一轮ID1基本固定的12组蓝圈样本',
    'snapshot_folder':'inputs',
    'manifest_sha256':hashlib.sha256((source/'inputs/samples.json').read_bytes()).hexdigest(),
    'latest_batch_original_rms_px':35.77146730655438,
    'latest_batch_best_tested_rms_px':25.97506692123237,
    'selection':'lowest all-sample reprojection RMS among tested candidates from the two batches'}
result['assessment'].update(weakly_excited_joint_ids=[1,2],weak_joint_directions_not_identified=True)
for path in [source/'hand_eye.json',HERE/'hand_eye.json']:
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if target.exists():
    raise FileExistsError('Final package already exists; do not overwrite it silently')
shutil.copytree(source,target)
X=np.array(result['T_gripper_camera'])
image=np.full((760,1350,3),255,np.uint8)
lines=['FINAL SELECTED HAND-EYE CALIBRATION',
       'Dataset: previous 12 blue-marker samples (ID1 nearly fixed)',
       f"Reprojection RMS: {result['rms_px_selected']:.4f} px",
       f"Position scatter: {result['board_origin_scatter_rms_mm']:.2f} mm; orientation scatter: {result['board_orientation_scatter_rms_deg']:.2f} deg",
       'T_gripper_camera (mm):',
       *['  '+'  '.join(f'{value:10.5f}' for value in row) for row in X],
       f"Fitted optical-origin distance: {result['camera_origin_distance_to_gripper_mm']:.2f} mm",
       'Use the embedded empirical joint mapping with this matrix.',
       'Training-data fit only; not independently measured physical accuracy.',
       'Weakly excited joints: ID1 and ID2; directions not identified by this batch.']
for index,line in enumerate(lines):
    cv2.putText(image,line,(25,40+42*index),cv2.FONT_HERSHEY_SIMPLEX,.59,(20,20,20),1,cv2.LINE_AA)
ok,buffer=cv2.imencode('.png',image)
if not ok:raise RuntimeError('Cannot encode report')
for path in [source/'hand_eye.png',target/'hand_eye.png',HERE/'hand_eye.png']:
    path.write_bytes(buffer.tobytes())
print('Packaged:',target)
