"""Audit the archived 97.93 px dataset without touching new captures."""
from pathlib import Path
import sys
import itertools
import json
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import read_json, read_image, board_points, chessboard_corners, camera_target_pose, arm_model, write_image
from part2_hand_eye_calibration.hand_eye import _fk_pose


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))


def main():
    archive=HERE/'samples/deleted_records'
    original=archive/'1792141637955515600'
    result=read_json(original/'previous_result/hand_eye.json')
    selected={}
    for folder in sorted(archive.iterdir()):
        if folder.name < original.name:
            continue
        manifest=folder/'records.json'
        if manifest.exists():
            for record in read_json(manifest)['samples']:
                if record['name'] in result['samples'] and record['name'] not in selected:
                    selected[record['name']] = (record,folder/'images'/record['image'])
    if set(selected)!=set(result['samples']):
        raise ValueError('Could not reconstruct the complete archived dataset')
    model=arm_model()
    intr=read_json(HERE.parent/'part1_camera_calibration/camera_intrinsics.json')
    K=np.array(intr['camera_matrix']); dist=np.array(intr['dist_coeffs'])
    points=board_points(8,6,25)
    rows=[]; frames=[]; poses=[]; cameras=[]; corner_sets=[]
    for name in result['samples']:
        item,path=selected[name]
        image=read_image(path); corners=chessboard_corners(image,8,6)
        camera=camera_target_pose(points,corners,K,dist)
        angles=model.angles(item['raw']); pose=_fk_pose(model,angles)
        rv=cv2.Rodrigues(camera[:3,:3])[0]
        projected,_=cv2.projectPoints(points,rv,camera[:3,3],K,dist)
        rows.append(dict(name=name,source=str(path),raw=item['raw'],angles_deg=np.degrees(angles).tolist(),
            pnp_rms_px=float(np.sqrt(np.mean(np.sum((projected-corners)**2,axis=2)))),
            pnp_rvec=rv.ravel().tolist()))
        cv2.drawChessboardCorners(image,(8,6),corners,True)
        cv2.circle(image,tuple(corners[0,0].astype(int)),8,(255,0,255),-1)
        cv2.putText(image,name,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        frames.append(cv2.resize(image,(320,240)))
        poses.append(pose); cameras.append(camera); corner_sets.append(corners)
    pairs=[]
    flip=np.diag([-1.,-1.,1.])
    for i,j in itertools.combinations(range(len(rows)),2):
        robot=angle(poses[i][:3,:3].T@poses[j][:3,:3])
        camera=angle(cameras[i][:3,:3]@cameras[j][:3,:3].T)
        alternate=angle(cameras[i][:3,:3]@flip@cameras[j][:3,:3].T)
        pairs.append(dict(a=rows[i]['name'],b=rows[j]['name'],robot_deg=robot,camera_deg=camera,
            camera_deg_if_one_board_order_reversed=alternate,disagreement_deg=abs(robot-camera)))
    output=dict(archived_result_rms_px=result['rms_px_selected'],samples=rows,pairs=pairs,
                current_model_zero_unwrapped=model.zero.tolist(),current_model_reference_deg=np.degrees(model.reference).tolist())
    folder=HERE/'failure_audit_20261016';folder.mkdir(exist_ok=True)
    (folder/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    write_image(folder/'contact_sheet.jpg',np.vstack([np.hstack(frames[i:i+4]) for i in range(0,12,4)]))
    for row in rows: print(row['name'],row['raw'],'q',np.round(row['angles_deg'],2).tolist())
    print('CONSECUTIVE')
    for p in pairs:
        if int(p['b'][-3:])-int(p['a'][-3:])==1:print(p)


if __name__=='__main__':main()
