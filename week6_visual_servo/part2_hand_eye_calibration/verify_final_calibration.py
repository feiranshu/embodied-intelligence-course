"""Independently reproject the packaged calibration using its embedded FK mapping."""
from pathlib import Path
import sys
import argparse
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import (read_json,write_json,arm_model,calibrated_gripper_pose,
                    calibration,board_points,inverse)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path,nargs='?',default=HERE/'最终标定结果')
    folder=parser.parse_args().folder
    result=read_json(folder/'hand_eye.json')
    manifest=read_json(folder/'inputs/samples.json')
    model=arm_model(folder/'inputs/advanced.json')
    K,dist,_=calibration(folder/'inputs/camera_intrinsics.json')
    X=np.array(result['T_gripper_camera']);Y=np.array(result['T_base_board'])
    points=board_points(8,6,25);squared=[];per_image={};depth=[]
    for sample in manifest['samples']:
        G=calibrated_gripper_pose(sample['raw'],model,result)
        C=inverse(G@X)@Y
        projected,_=cv2.projectPoints(points,cv2.Rodrigues(C[:3,:3])[0],C[:3,3],K,dist)
        corners=np.array(sample['ordered_corners'],np.float32).reshape(-1,1,2)
        error=np.sum((projected-corners)**2,axis=2).ravel()
        squared.extend(error.tolist());per_image[sample['name']]=float(np.sqrt(np.mean(error)))
        depth.append(float(np.min((points@C[:3,:3].T+C[:3,3])[:,2])))
    rms=float(np.sqrt(np.mean(squared)))
    report=dict(recomputed_rms_px=rms,saved_rms_px=result['rms_px_selected'],
                sample_count=len(manifest['samples']),per_image_rms_px=per_image,
                rotation_determinant=float(np.linalg.det(X[:3,:3])),
                rotation_orthogonality_error=float(np.max(np.abs(X[:3,:3].T@X[:3,:3]-np.eye(3)))),
                all_board_depths_positive=all(d>0 for d in depth),
                independent_physical_accuracy_measured=False)
    if 'distance_measurement' in result:
        report['distance_is_input_assumption']=True
        report['assumed_distance_mm']=result['distance_measurement']['fit_radius_mm']
        report['matrix_distance_mm']=float(np.linalg.norm(X[:3,3]))
        if abs(report['matrix_distance_mm']-report['assumed_distance_mm'])>1e-6:raise ValueError(report)
    if abs(rms-result['rms_px_selected'])>.001:raise ValueError(report)
    if report['rotation_orthogonality_error']>1e-8 or not report['all_board_depths_positive']:raise ValueError(report)
    write_json(folder/'verification.json',report)
    print(report)


if __name__=='__main__':main()
