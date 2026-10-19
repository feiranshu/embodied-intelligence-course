"""Read-only offline audit of saved images, intrinsics, and Week 4 FK."""
from pathlib import Path
import json
import sys
import itertools
import xml.etree.ElementTree as ET
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WEEK6 = HERE.parent
ARM = WEEK6.parent / 'week4' / 'so101_scs215_portable'
sys.path[:0] = [str(WEEK6), str(ARM)]
from common import read_json, read_image, chessboard_corners, board_points, camera_target_pose
from advanced import Kinematics
from hand_eye import _fk_pose


def rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def main():
    intr = read_json(WEEK6 / 'part1_camera_calibration' / 'camera_intrinsics.json')
    K = np.array(intr['camera_matrix']); dist = np.array(intr['dist_coeffs'])
    points = board_points(8, 6, 25.)
    model = Kinematics(read_json(ARM / 'advanced.json'))
    output = {'intrinsics_saved_rms_px': intr['rms_px'], 'square_mm': 25.,
              'fk_source': str(ARM), 'samples': [], 'pairs': []}
    original_corners = []
    for path in sorted((WEEK6 / 'part1_camera_calibration' / 'evidence' / 'task1_handoff').glob('intrinsic_*.png')):
        corners = chessboard_corners(read_image(path), 8, 6)
        if corners is not None:
            original_corners.append(corners)
    rms, newK, newdist, _, _ = cv2.calibrateCamera([points]*len(original_corners), original_corners, (640,480), None, None)
    output['intrinsics_redetection'] = {'count': len(original_corners), 'rms_px': rms,
                                      'K': newK.tolist(), 'dist': newdist.tolist()}
    fixed_errors = []
    for corners in original_corners:
        ok, rv, tv = cv2.solvePnP(points, corners, K, dist)
        predicted, _ = cv2.projectPoints(points, rv, tv, K, dist)
        fixed_errors.append(float(np.sqrt(np.mean(np.sum((predicted-corners)**2,axis=2)))))
    output['intrinsics_saved_K_redetected_rms_px'] = float(np.sqrt(np.mean(np.square(fixed_errors))))
    output['intrinsics_saved_K_per_image_px'] = fixed_errors
    poses = []; cameras = []; corners_all = []; angles_all = []
    document = ET.parse(ARM / 'so101_new_calib.urdf')
    joints = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper_frame_joint']
    max_fk_delta = 0.
    # Independent homogeneous URDF traversal using Rodrigues for arbitrary joint axes.
    for sample in read_json(HERE / 'samples' / 'samples.json')['samples']:
        frame = read_image(HERE / 'samples' / sample['image'])
        corners = chessboard_corners(frame, 8, 6)
        Tct = camera_target_pose(points, corners, K, dist)
        angles = model.angles(sample['raw']); Tbg = _fk_pose(model, angles)
        independent = np.eye(4)
        for idx, name in enumerate(joints):
            joint = document.find(f"joint[@name='{name}']"); origin = joint.find('origin')
            r,p,y = np.fromstring(origin.get('rpy'), sep=' ')
            Rx = cv2.Rodrigues(np.array([r,0.,0.]))[0]
            Ry = cv2.Rodrigues(np.array([0.,p,0.]))[0]
            Rz = cv2.Rodrigues(np.array([0.,0.,y]))[0]
            O = np.eye(4); O[:3,:3] = Rz@Ry@Rx
            O[:3,3] = np.fromstring(origin.get('xyz'),sep=' ')*1000
            independent = independent@O
            if idx < 5:
                axis = np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                J = np.eye(4); J[:3,:3] = cv2.Rodrigues(axis*angles[idx])[0]
                independent = independent@J
        max_fk_delta = max(max_fk_delta, float(np.max(np.abs(independent-Tbg))))
        rv = cv2.Rodrigues(Tct[:3,:3])[0]
        predicted,_ = cv2.projectPoints(points,rv,Tct[:3,3],K,dist)
        rms = float(np.sqrt(np.mean(np.sum((predicted-corners)**2,axis=2))))
        xy = corners.reshape(-1,2)
        output['samples'].append({'name': sample['name'], 'raw':sample['raw'],
          'angles_deg': np.degrees(angles).tolist(), 'fk_xyz_mm':Tbg[:3,3].tolist(),
          'pnp_rms_px':rms, 'board_depth_mm':float(Tct[2,3]),
          'board_width_px':float(np.linalg.norm(xy[7]-xy[0])),
          'board_center_px':xy.mean(axis=0).tolist()})
        poses.append(Tbg); cameras.append(Tct); corners_all.append(corners); angles_all.append(angles)
    output['independent_urdf_max_matrix_difference'] = max_fk_delta
    output['joint_angle_span_deg'] = np.ptp(np.degrees(angles_all),axis=0).tolist()
    for i,j in itertools.combinations(range(len(poses)),2):
        robot = rotation_angle(poses[i][:3,:3].T@poses[j][:3,:3])
        camera = rotation_angle(cameras[i][:3,:3]@cameras[j][:3,:3].T)
        output['pairs'].append({'a':output['samples'][i]['name'],'b':output['samples'][j]['name'],
          'robot_rotation_deg':robot,'camera_rotation_deg':camera,
          'rotation_disagreement_deg':abs(robot-camera),
          'corner_displacement_rms_px':float(np.sqrt(np.mean(np.sum((corners_all[i]-corners_all[j])**2,axis=2)))),
          'raw_delta':(np.array(output['samples'][j]['raw'])-output['samples'][i]['raw']).tolist()})
    # Sensitivity experiment, not a measured confidence interval: perturb detected corners by 0.2 px.
    rng = np.random.default_rng(20261016)
    sensitivity = []
    for idx in [2,5]:
        errors = []
        for _ in range(250):
            noisy = corners_all[idx].astype(float)+rng.normal(0,.2,corners_all[idx].shape)
            perturbed = camera_target_pose(points,noisy,K,dist)
            errors.append(rotation_angle(perturbed[:3,:3]@cameras[idx][:3,:3].T))
        sensitivity.append({'name':output['samples'][idx]['name'],
          'assumed_corner_noise_sigma_px':.2,'median_rotation_change_deg':float(np.median(errors)),
          'p95_rotation_change_deg':float(np.percentile(errors,95))})
    output['pnp_noise_sensitivity'] = sensitivity
    path = HERE / 'dataset_diagnosis.json'
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    summary = {k:v for k,v in output.items() if k not in ['samples','pairs','intrinsics_saved_K_per_image_px']}
    summary['selected_pair'] = next(p for p in output['pairs'] if p['a']=='pose_004' and p['b']=='pose_007')
    summary['pnp_rms_range_px'] = [min(s['pnp_rms_px'] for s in output['samples']),max(s['pnp_rms_px'] for s in output['samples'])]
    summary['board_width_range_px'] = [min(s['board_width_px'] for s in output['samples']),max(s['board_width_px'] for s in output['samples'])]
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
