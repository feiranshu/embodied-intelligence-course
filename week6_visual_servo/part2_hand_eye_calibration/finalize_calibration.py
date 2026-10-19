"""Compare empirical mappings offline and save a reproducible best-fit calibration."""
from pathlib import Path
import sys
import math
import time
import shutil
import itertools
import json
import hashlib
import argparse
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import (ARM_CODE,read_json,write_json,board_points,board_spec,calibration,
                    read_image,arm_model,inverse,transform,write_image)
from board_orientation import sample_board_corners
from part2_hand_eye_calibration.hand_eye import (_fk_pose,_closed_form,_refine_hand_eye,
                                                _pose_from_vector,_pose_to_vector,HAND_EYE_METHODS)


def refine_mapping(obs,model,K,dist,points,rates,raw_center,reference,seed,fit_scales):
    spans=np.ptp(np.array([item['unwrapped'][:5] for item in obs]),axis=0)
    variable=[i for i in range(5) if spans[i]>=10] if fit_scales else []
    p=np.concatenate([*_pose_to_vector(seed['camera']),*_pose_to_vector(seed['board']),
                      seed.get('delta',np.zeros(5))[1:4],np.zeros(len(variable))])
    base_rates=np.asarray(rates,float)
    def unpack(v):
        rate=base_rates.copy()
        if variable:rate[variable]*=np.exp(np.clip(v[15:],-5,5))
        offset=np.zeros(5);offset[1:4]=v[12:15]
        X=_pose_from_vector(v[:3],v[3:6]);Y=_pose_from_vector(v[6:9],v[9:12])
        return X,Y,rate,offset
    def residual(v):
        X,Y,rate,offset=unpack(v)
        values=[]
        for item in obs:
            q=reference+(item['unwrapped'][:5]-raw_center)*np.radians(rate)+offset
            C=inverse(_fk_pose(model,q)@X)@Y
            if C[2,3]<=1:
                values.append(np.full(item['corners'].size,1e4));continue
            projected=cv2.projectPoints(points,cv2.Rodrigues(C[:3,:3])[0],C[:3,3],K,dist)[0]
            values.append((projected-item['corners']).ravel())
        return np.concatenate(values)
    steps=np.array([1e-6]*3+[1e-4]*3+[1e-6]*3+[1e-4]*3+[1e-6]*3+[1e-5]*len(variable))
    e=residual(p);cost=float(e@e);damping=.001
    for iteration in range(220):
        J=np.column_stack([(residual(p+np.eye(len(p))[i]*steps[i])-e)/steps[i] for i in range(len(p))])
        normal=J.T@J;gradient=J.T@e;improved=False;previous=cost
        while damping<1e12:
            try:change=-np.linalg.solve(normal+damping*np.diag(np.diag(normal)+1e-12),gradient)
            except np.linalg.LinAlgError:damping*=4;continue
            trial=p+change
            if variable:
                # Search bounds on fitted model parameters, not additional capture conditions.
                adjusted=np.abs(base_rates[variable])*np.exp(np.clip(trial[15:],-5,5))
                if np.any(adjusted<.05) or np.any(adjusted>1.):damping*=4;continue
            error=residual(trial)
            with np.errstate(over='ignore',invalid='ignore'):newcost=float(error@error)
            if np.isfinite(newcost) and newcost<cost:
                p=trial;e=error;cost=newcost;damping=max(damping/3,1e-12);improved=True;break
            damping*=4
        if not improved or previous-cost<1e-9*max(previous,1):break
    X,Y,rate,offset=unpack(p)
    return dict(camera=X,board=Y,rates=rate,reference=reference+offset,
                rms_px=float(np.sqrt(cost/(e.size/2))),
                per_image_rms_px=[float(np.sqrt(np.mean(x*x)*2)) for x in np.split(e,len(obs))],
                iterations=iteration+1,fit_scales=fit_scales)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=HERE/'final_calibration_20261016_174100')
    parser.add_argument('--snapshot-only',action='store_true')
    args=parser.parse_args();folder=args.folder
    snapshot=folder/'inputs'
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for source in [HERE/'samples/samples.json',HERE/'hand_eye.json',
                       HERE.parent/'part1_camera_calibration/camera_intrinsics.json',ARM_CODE/'advanced.json',
                       ARM_CODE/'so101_new_calib.urdf']:
            shutil.copy2(source,snapshot/source.name)
        if (HERE/'hand_eye.png').exists():shutil.copy2(HERE/'hand_eye.png',snapshot/'hand_eye.png')
        (snapshot/'images').mkdir()
        for item in read_json(snapshot/'samples.json')['samples']:
            path=(HERE/'samples'/item['image']).resolve()
            if path.parent!=(HERE/'samples').resolve():raise ValueError('Unexpected image path')
            shutil.copy2(path,snapshot/'images'/path.name)
    if args.snapshot_only:return
    original=read_json(snapshot/'hand_eye.json');manifest=read_json(snapshot/'samples.json')
    intr=read_json(snapshot/'camera_intrinsics.json');K,dist,size=calibration(snapshot/'camera_intrinsics.json')
    cols,rows,square=board_spec(intr);points=board_points(cols,rows,square)
    model=arm_model(snapshot/'advanced.json');obs=[]
    for item in manifest['samples']:
        frame=read_image(snapshot/'images'/Path(item['image']).name)
        corners=sample_board_corners(frame,item,cols,rows)
        if corners is None:raise ValueError(f"Missing board: {item['name']}")
        obs.append(dict(name=item['name'],raw=np.array(item['raw']),unwrapped=model.unwrap(item['raw']),
                        corners=corners,T_cm=None))
    from common import camera_target_pose
    for item in obs:item['T_cm']=camera_target_pose(points,item['corners'],K,dist)
    center=np.median(np.array([x['unwrapped'][:5] for x in obs]),axis=0)
    # Preserve original URDF reference convention at the new numerical anchor.
    reference=(center-model.zero)*model.signs*model.scale+model.reference
    outline=points[[0,cols-1,cols*rows-1,cols*(rows-1)]]
    seeds=[]
    rate_sets=[[300/1023]*5,[360/1023]*5,[300/1023,300/1023,.38803755,.40914280,.34524328]]
    for signs in itertools.product([-1,1],repeat=3):
        base,shoulder,pitch=signs
        directions=np.array([base,shoulder,pitch,pitch,1])
        for gains in rate_sets:
            rates=np.array(gains)*directions
            observations=[dict(item,angles=reference+(item['unwrapped'][:5]-center)*np.radians(rates)) for item in obs]
            attempts=[]
            for method_name in ['PARK','HORAUD','ANDREFF','DANIILIDIS']:
                try:
                    X,Y,spread,_=_closed_form(observations,model,np.zeros(5),outline,HAND_EYE_METHODS[method_name])
                    if np.isfinite(X).all():attempts.append(dict(camera=X,board=Y,spread=spread,method=method_name))
                except (ValueError,cv2.error):pass
            if attempts:
                chosen=min(attempts,key=lambda v:v['spread'])
                seeds.append(dict(chosen,rates=rates,observations=observations))
    # Initial ranking only seeds the nonlinear search; final ranking uses all-corner RMS.
    seeds.sort(key=lambda x:x['spread'])
    candidates=[];solutions=[]
    baseline_seed=dict(camera=np.array(original['T_gripper_camera']),board=np.array(original['T_base_board']),
                       delta=np.radians(original['joint_offsets_deg']))
    baseline_rates=model.signs*math.degrees(model.scale)
    current=refine_mapping(obs,model,K,dist,points,baseline_rates,center,reference,baseline_seed,True)
    current['method']='original_result_seed';solutions.append(current)
    print('REFINED ORIGINAL',current['rms_px'],current['rates'].tolist(),flush=True)
    for i,seed in enumerate(seeds[:14]):
        try:
            fit=_refine_hand_eye(seed['observations'],model,K,dist,points,seed['camera'],seed['board'],[1,2,3])
            fit['method']=seed['method'];fit['rates']=seed['rates']
            candidates.append(fit)
            print('FIXED',i,seed['method'],fit['rms_px'],seed['rates'].tolist(),flush=True)
        except (ValueError,cv2.error,np.linalg.LinAlgError) as error:print('FAILED',i,str(error),flush=True)
    candidates.sort(key=lambda x:x['rms_px'])
    for i,seed in enumerate(candidates[:6]):
        fit=refine_mapping(obs,model,K,dist,points,seed['rates'],center,reference,seed,True)
        fit['method']=seed['method'];solutions.append(fit)
        print('VARIABLE',i,fit['rms_px'],fit['rates'].tolist(),flush=True)
    for candidate in candidates:
        solutions.append(dict(camera=candidate['camera'],board=candidate['board'],rates=candidate['rates'],
            reference=reference+candidate['delta'],rms_px=candidate['rms_px'],per_image_rms_px=candidate['per_image_rms_px'],
            method=candidate['method'],fit_scales=False,iterations=None))
    best=min(solutions,key=lambda x:x['rms_px'])
    X,Y=best['camera'],best['board'];boards=[]
    for item in obs:
        q=best['reference']+(item['unwrapped'][:5]-center)*np.radians(best['rates'])
        boards.append(_fk_pose(model,q)@X@item['T_cm'])
    positions=np.array([p[:3,3] for p in boards]);u,_,vt=np.linalg.svd(sum(p[:3,:3] for p in boards))
    meanR=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    angles=[math.degrees(math.acos(float(np.clip((np.trace(meanR.T@p[:3,:3])-1)/2,-1,1)))) for p in boards]
    result=dict(original)
    result.update(T_gripper_camera=X.tolist(),T_base_board=Y.tolist(),estimated_board_origin_base_mm=Y[:3,3].tolist(),
        camera_translation_in_gripper_mm=X[:3,3].tolist(),camera_origin_distance_to_gripper_mm=float(np.linalg.norm(X[:3,3])),
        board_origin_scatter_rms_mm=float(np.sqrt(np.mean(np.sum((positions-positions.mean(axis=0))**2,axis=1)))),
        board_orientation_scatter_rms_deg=float(np.sqrt(np.mean(np.square(angles)))),
        rms_px_nominal_fk=original['rms_px_nominal_fk'],rms_px_with_joint_offsets=best['rms_px'],rms_px_selected=best['rms_px'],
        per_image_rms_px=dict(zip([o['name'] for o in obs],best['per_image_rms_px'])),
        selected_hand_eye_method=best['method']+' + empirical mapping refinement',
        joint_offsets_deg=np.degrees(best['reference']-reference).tolist(),
        calibrated_joint_mapping=dict(raw_reference_unwrapped=center.tolist(),angle_reference_deg=np.degrees(best['reference']).tolist(),
            signed_degrees_per_raw=best['rates'].tolist(),unwrap_config='inputs/advanced.json'),
        empirical_mapping_required=True,
        assessment=dict(status='best_fit_on_this_dataset',independent_accuracy_validated=False,
            camera_distance_independently_measured=False,training_samples=len(obs),all_samples_used=True,
            note='Use the embedded joint mapping with this matrix. Training fit is not independent physical validation.'),
        original_unmodified_fit_rms_px=original['rms_px_selected'])
    result.pop('kinematics_patch_suggestion',None)
    result.pop('solver_candidates',None)
    result.pop('joint_offset_std_deg',None);result.pop('unobservable_offset_joint_ids',None)
    R=X[:3,:3]
    result['camera_rpy_deg_in_gripper']=np.degrees([math.atan2(R[2,1],R[2,2]),math.atan2(-R[2,0],math.hypot(R[0,0],R[1,0])),math.atan2(R[1,0],R[0,0])]).tolist()
    rows=[]
    for sol in sorted(solutions,key=lambda x:x['rms_px']):
        rows.append(dict(method=sol['method'],rms_px=sol['rms_px'],signed_degrees_per_raw=sol['rates'].tolist(),
                         fitted_scales=sol['fit_scales'],distance_mm=float(np.linalg.norm(sol['camera'][:3,3]))))
    write_json(folder/'candidate_results.json',{'candidates':rows,'selection':'minimum all-sample reprojection RMS'})
    write_json(folder/'hand_eye.json',result)
    # Write the familiar result after backing up inputs; preserve motor-control configuration.
    write_json(HERE/'hand_eye.json',result)
    report=np.full((720,1250,3),255,np.uint8)
    lines=['FINAL BEST-FIT HAND-EYE CALIBRATION',f"Samples: {len(obs)}; RMS: {best['rms_px']:.4f} px",
           f"Original result: {original['rms_px_selected']:.4f} px",'T_gripper_camera (mm):',
           *['  '+'  '.join(f'{v:10.5f}' for v in row) for row in X],
           f"Fitted origin distance: {np.linalg.norm(X[:3,3]):.2f} mm",
           'Signed degree/count: '+str(np.round(best['rates'],6).tolist()),
           'EMPIRICAL JOINT MAPPING REQUIRED; see JSON.',
           'Training-data fit only. Independent accuracy not measured.']
    for i,line in enumerate(lines):cv2.putText(report,line,(25,40+40*i),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
    write_image(folder/'hand_eye.png',report);write_image(HERE/'hand_eye.png',report)
    print('FINAL',json.dumps({'rms_px':best['rms_px'],'distance_mm':float(np.linalg.norm(X[:3,3])),
                              'rates':best['rates'].tolist(),'folder':str(folder)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
