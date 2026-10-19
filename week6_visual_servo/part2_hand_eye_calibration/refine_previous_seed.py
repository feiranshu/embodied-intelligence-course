"""Include the previously reported 9.22 px candidate as an explicit finalization seed."""
from pathlib import Path
import sys
import math
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import *
from board_orientation import sample_board_corners
from part2_hand_eye_calibration.hand_eye import _fk_pose
from part2_hand_eye_calibration.finalize_calibration import refine_mapping


def main():
    folder=HERE/'final_calibration_20261016_previous';snapshot=folder/'inputs'
    model=arm_model(snapshot/'advanced.json');K,dist,_=calibration(snapshot/'camera_intrinsics.json')
    points=board_points(8,6,25);obs=[]
    for item in read_json(snapshot/'samples.json')['samples']:
        frame=read_image(snapshot/'images'/item['image']);corners=sample_board_corners(frame,item,8,6)
        obs.append(dict(name=item['name'],raw=item['raw'],unwrapped=model.unwrap(item['raw']),
                        corners=corners,T_cm=camera_target_pose(points,corners,K,dist)))
    center=np.median(np.array([x['unwrapped'][:5] for x in obs]),axis=0)
    prior=read_json(HERE/'stationary_id1_audit_20261016/mapping_sensitivity_results.json')
    seeds=[]
    for branch in prior['branches']:
        for candidate in branch['candidates']:
            seeds.append(dict(candidate,rates=np.array([300/1023,-300/1023,*branch['signed_degrees_per_count']])))
    seeds.sort(key=lambda x:x['rms_px']);outcomes=[]
    for item in seeds[:3]:
        rates=item['rates'];reference=model.reference+(center-model.zero)*np.radians(rates)
        offset=np.radians(item['offsets_deg']);X=np.array(item['T_gripper_camera'])
        boards=[]
        for o in obs:
            q=reference+(o['unwrapped'][:5]-center)*np.radians(rates)+offset
            boards.append(_fk_pose(model,q)@X@o['T_cm'])
        u,_,vt=np.linalg.svd(sum(x[:3,:3] for x in boards))
        Y=transform(u@np.diag([1,1,np.linalg.det(u@vt)])@vt,np.mean([x[:3,3] for x in boards],axis=0))
        fitted=refine_mapping(obs,model,K,dist,points,rates,center,reference,dict(camera=X,board=Y,delta=offset),True)
        fitted['method']='previous_'+item['method']+'_seed'
        outcomes.append(fitted)
        print(fitted['method'],fitted['rms_px'],fitted['rates'].tolist(),flush=True)
    old=read_json(folder/'hand_eye.json');best=min(outcomes,key=lambda x:x['rms_px'])
    write_json(folder/'previous_seed_comparison.json',{'candidates':[{'method':x['method'],'rms_px':x['rms_px'],
        'rates':x['rates'].tolist()} for x in outcomes]})
    if best['rms_px']>=old['rms_px_selected']:return
    X,Y=best['camera'],best['board'];boards=[]
    for o in obs:
        q=best['reference']+(o['unwrapped'][:5]-center)*np.radians(best['rates'])
        boards.append(_fk_pose(model,q)@X@o['T_cm'])
    pos=np.array([x[:3,3] for x in boards]);u,_,vt=np.linalg.svd(sum(x[:3,:3] for x in boards));Rmean=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    angles=[math.degrees(math.acos(float(np.clip((np.trace(Rmean.T@x[:3,:3])-1)/2,-1,1)))) for x in boards]
    old.update(T_gripper_camera=X.tolist(),T_base_board=Y.tolist(),estimated_board_origin_base_mm=Y[:3,3].tolist(),
        camera_translation_in_gripper_mm=X[:3,3].tolist(),camera_origin_distance_to_gripper_mm=float(np.linalg.norm(X[:3,3])),
        board_origin_scatter_rms_mm=float(np.sqrt(np.mean(np.sum((pos-pos.mean(axis=0))**2,axis=1)))),
        board_orientation_scatter_rms_deg=float(np.sqrt(np.mean(np.square(angles)))),
        rms_px_with_joint_offsets=best['rms_px'],rms_px_selected=best['rms_px'],
        per_image_rms_px=dict(zip([o['name'] for o in obs],best['per_image_rms_px'])),
        selected_hand_eye_method=best['method']+' + empirical mapping refinement',
        joint_offsets_deg=np.degrees(best['reference']-((center-model.zero)*model.signs*model.scale+model.reference)).tolist())
    old['calibrated_joint_mapping'].update(raw_reference_unwrapped=center.tolist(),angle_reference_deg=np.degrees(best['reference']).tolist(),signed_degrees_per_raw=best['rates'].tolist())
    R=X[:3,:3];old['camera_rpy_deg_in_gripper']=np.degrees([math.atan2(R[2,1],R[2,2]),math.atan2(-R[2,0],math.hypot(R[0,0],R[1,0])),math.atan2(R[1,0],R[0,0])]).tolist()
    write_json(folder/'hand_eye.json',old);write_json(HERE/'hand_eye.json',old)
    print('UPDATED',best['rms_px'],flush=True)


if __name__=='__main__':main()
