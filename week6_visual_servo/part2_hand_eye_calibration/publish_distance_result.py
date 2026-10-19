"""Publish the verified distance-informed result, preserving the previous package."""
from pathlib import Path
import shutil
import hashlib
import numpy as np
import cv2
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import read_json,write_json,write_image


def main():
    fits=HERE/'final_calibration_20261016_distance120'
    comparison=read_json(fits/'comparison.json')
    source=fits/comparison['selected']
    result=read_json(source/'hand_eye.json')
    verification=read_json(source/'verification.json')
    if abs(verification['recomputed_rms_px']-result['rms_px_selected'])>.001:
        raise ValueError('Independent verification mismatch')
    if abs(np.linalg.norm(np.array(result['T_gripper_camera'])[:3,3])-120)>1e-6:
        raise ValueError('Unexpected distance')
    result['calibration_dataset'].update(
        manifest_sha256=hashlib.sha256((source/'inputs/samples.json').read_bytes()).hexdigest(),
        selection='minimum tested all-corner RMS at the same 120 mm user distance assumption',
        comparison_file='comparison.json')
    write_json(source/'hand_eye.json',result)
    target=HERE/'最终标定结果'
    archive=HERE/'最终标定结果_旧版未用实测距离'
    if target.exists():
        if archive.exists():raise FileExistsError('Archive already exists; inspect before publishing again')
        # Check both resolved paths before moving the complete directory.
        for path in [target,archive]:
            path.resolve().relative_to(HERE.resolve())
            if path.resolve()==HERE.resolve():raise ValueError('Invalid archive path')
        target.rename(archive)
        note='> 此为旧版376 mm拟合结果，已由采用约12 cm测量依据的新版本替代。当前结果见旁边的“最终标定结果”文件夹。\n\n'
        old=archive/'README.md'
        old.write_text(note+old.read_text(encoding='utf-8'),encoding='utf-8')
        for name in ['hand_eye.json','hand_eye.png']:
            if (HERE/name).exists():shutil.copy2(HERE/name,archive/('previous_root_'+name))
    shutil.copytree(source,target)
    shutil.copy2(fits/'comparison.json',target/'comparison.json')
    shutil.copy2(fits/'README_final.md',target/'README.md')
    X=np.array(result['T_gripper_camera'])
    report=np.full((760,1400,3),255,np.uint8)
    lines=['FINAL HAND-EYE: USER DISTANCE ASSUMPTION',
        'Dataset: previous 12 blue-marker samples (ID1 nearly fixed)',
        'Camera-to-gripper origin distance: 120 mm FIXED FROM APPROXIMATE USER MEASUREMENT',
        f"All-corner RMS: {result['rms_px_selected']:.4f} px; all 12 samples included",
        f"Board scatter: {result['board_origin_scatter_rms_mm']:.2f} mm; {result['board_orientation_scatter_rms_deg']:.2f} deg",
        'T_gripper_camera (translation mm):',
        *['  '+'  '.join(f'{x:11.6f}' for x in row) for row in X],
        'Distance is a fitting input; not independently recovered or verified.',
        'Embedded empirical FK mapping required. ID1/ID2 weakly excited.',
        'Independent physical positioning accuracy has not been measured.']
    for i,line in enumerate(lines):
        cv2.putText(report,line,(25,40+43*i),cv2.FONT_HERSHEY_SIMPLEX,.59,(20,20,20),1,cv2.LINE_AA)
    for folder in [source,target,HERE]:
        write_json(folder/'hand_eye.json',result)
        write_image(folder/'hand_eye.png',report)
    print('Published:',target)


if __name__=='__main__':main()
