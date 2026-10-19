"""Offline hand-eye fit with a user-measured approximate camera offset length.

The supplied distance is a fitting assumption, never an independent validation.
No capture rules, servo configuration, or physical robot state are changed.
"""
from pathlib import Path
import argparse
import itertools
import math
import shutil
import sys
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (read_json, write_json, arm_model, calibration, board_points,
                    board_spec, inverse, camera_target_pose, transform)
from part2_hand_eye_calibration.hand_eye import (
    _fk_pose, _closed_form, _refine_hand_eye, _pose_to_vector,
    _pose_from_vector, HAND_EYE_METHODS)


def fit_radius(obs, model, K, dist, points, center, reference, rates, seed, radius, fit_scales):
    direction = seed['camera'][:3, 3]
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.array([0., 0., 1.]); norm = 1.
    azimuth = math.atan2(direction[1], direction[0])
    elevation = math.asin(float(np.clip(direction[2] / norm, -1, 1)))
    span = np.ptp(np.array([x['unwrapped'][:5] for x in obs]), axis=0)
    variable = [i for i in range(5) if span[i] >= 10] if fit_scales else []
    p = np.concatenate([_pose_to_vector(seed['camera'])[0], [azimuth, elevation],
                        *_pose_to_vector(seed['board']), seed.get('delta', np.zeros(5))[1:4],
                        np.zeros(len(variable))])
    base_rates = np.array(rates, float)

    def unpack(v):
        az, el = v[3:5]
        t = radius * np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
        X = _pose_from_vector(v[:3], t)
        Y = _pose_from_vector(v[5:8], v[8:11])
        offset = np.zeros(5); offset[1:4] = v[11:14]
        rate = base_rates.copy()
        if variable: rate[variable] *= np.exp(np.clip(v[14:], -5, 5))
        return X, Y, offset, rate

    def residual(v):
        X, Y, offset, rate = unpack(v)
        values = []
        for item in obs:
            q = reference + offset + (item['unwrapped'][:5]-center)*np.radians(rate)
            C = inverse(_fk_pose(model, q) @ X) @ Y
            if C[2, 3] <= 1:
                values.append(np.full(item['corners'].size, 1e4)); continue
            projected = cv2.projectPoints(points, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], K, dist)[0]
            values.append((projected-item['corners']).ravel())
        return np.concatenate(values)

    steps = np.array([1e-6]*8 + [1e-4]*3 + [1e-6]*3 + [1e-5]*len(variable))
    e = residual(p); cost = float(e@e); damping = .001
    for iteration in range(220):
        J = np.column_stack([(residual(p+np.eye(len(p))[i]*steps[i])-e)/steps[i] for i in range(len(p))])
        normal = J.T@J; gradient = J.T@e; improved = False; previous = cost
        while damping < 1e12:
            try: change = -np.linalg.solve(normal+damping*np.diag(np.diag(normal)+1e-12), gradient)
            except np.linalg.LinAlgError: damping *= 4; continue
            trial = p + change
            if variable:
                adjusted = abs(base_rates[variable])*np.exp(np.clip(trial[14:], -5, 5))
                if np.any(adjusted < .05) or np.any(adjusted > 1.): damping *= 4; continue
            error = residual(trial)
            with np.errstate(over='ignore', invalid='ignore'): newcost = float(error@error)
            if np.isfinite(newcost) and newcost < cost:
                p = trial; e = error; cost = newcost; damping = max(damping/3, 1e-12); improved = True; break
            damping *= 4
        if not improved or previous-cost < 1e-9*max(previous, 1): break
    X, Y, offset, rate = unpack(p)
    return dict(camera=X, board=Y, rates=rate, reference=reference+offset,
                rms_px=float(np.sqrt(np.mean(e*e)*2)),
                per_image_rms_px=[float(np.sqrt(np.mean(x*x)*2)) for x in np.split(e, len(obs))],
                iterations=iteration+1, fitted_scales=fit_scales)


def prepare(folder):
    snapshot = folder/'inputs'
    manifest = read_json(snapshot/'samples.json')
    model = arm_model(snapshot/'advanced.json')
    K, dist, _ = calibration(snapshot/'camera_intrinsics.json')
    cols, rows, square = board_spec(read_json(snapshot/'camera_intrinsics.json'))
    points = board_points(cols, rows, square)
    obs = []
    for item in manifest['samples']:
        corners = np.array(item['ordered_corners'], np.float32).reshape(-1, 1, 2)
        obs.append(dict(name=item['name'], unwrapped=model.unwrap(item['raw']), corners=corners,
                        T_cm=camera_target_pose(points, corners, K, dist)))
    center = np.median(np.array([x['unwrapped'][:5] for x in obs]), axis=0)
    reference = (center-model.zero)*model.signs*model.scale+model.reference
    return model, K, dist, points, obs, center, reference


def result_json(folder, best, args, source_label):
    model, K, dist, points, obs, center, reference = prepare(folder)
    X, Y = best['camera'], best['board']
    boards = []
    for item in obs:
        q = best['reference']+(item['unwrapped'][:5]-center)*np.radians(best['rates'])
        boards.append(_fk_pose(model, q)@X@item['T_cm'])
    positions = np.array([T[:3, 3] for T in boards])
    u, _, vt = np.linalg.svd(sum(T[:3, :3] for T in boards))
    meanR = u@np.diag([1, 1, np.linalg.det(u@vt)])@vt
    angles = [math.degrees(math.acos(float(np.clip((np.trace(meanR.T@T[:3,:3])-1)/2, -1, 1)))) for T in boards]
    result = read_json(folder/'inputs/hand_eye.json')
    for key in ['kinematics_patch_suggestion', 'solver_candidates', 'joint_offset_std_deg', 'unobservable_offset_joint_ids']:
        result.pop(key, None)
    span = np.ptp(np.array([o['unwrapped'][:5] for o in obs]), axis=0)
    result.update(T_gripper_camera=X.tolist(), T_base_board=Y.tolist(),
        estimated_board_origin_base_mm=Y[:3,3].tolist(), camera_translation_in_gripper_mm=X[:3,3].tolist(),
        camera_origin_distance_to_gripper_mm=float(np.linalg.norm(X[:3,3])),
        board_origin_scatter_rms_mm=float(np.sqrt(np.mean(np.sum((positions-positions.mean(axis=0))**2, axis=1)))),
        board_orientation_scatter_rms_deg=float(np.sqrt(np.mean(np.square(angles)))),
        rms_px_selected=best['rms_px'], rms_px_with_joint_offsets=best['rms_px'],
        per_image_rms_px=dict(zip([o['name'] for o in obs], best['per_image_rms_px'])),
        selected_hand_eye_method=best['method']+' + measured-length-constrained empirical fit',
        joint_offsets_deg=np.degrees(best['reference']-reference).tolist(),
        calibrated_joint_mapping=dict(raw_reference_unwrapped=center.tolist(),
            angle_reference_deg=np.degrees(best['reference']).tolist(),
            signed_degrees_per_raw=best['rates'].tolist(), unwrap_config='inputs/advanced.json'),
        empirical_mapping_required=True,
        calibration_dataset=dict(label=source_label, snapshot_folder='inputs'),
        distance_measurement=dict(user_approximate_mm=args.distance_mm, fit_radius_mm=args.distance_mm,
            mode='fixed length assumed from approximate user measurement', uncertainty_mm=None,
            independently_validates_fitted_distance=False,
            frame_assumption='User approximate distance interpreted as optical-origin to gripper_frame_link origin'),
        assessment=dict(status='best_tested_fit_with_user_distance_assumption', independent_accuracy_validated=False,
            camera_distance_independently_measured=False, training_samples=len(obs), all_samples_used=True,
            weakly_excited_joint_ids=[i+1 for i in range(5) if span[i]<10],
            weak_joint_directions_not_identified=bool(np.any(span<10)),
            note='Distance is an input assumption, not an independently recovered result. Embedded empirical FK mapping required.'))
    R=X[:3,:3]
    result['camera_rpy_deg_in_gripper']=np.degrees([math.atan2(R[2,1],R[2,2]),math.atan2(-R[2,0],math.hypot(R[0,0],R[1,0])),math.atan2(R[1,0],R[0,0])]).tolist()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance-mm',type=float,default=120.)
    parser.add_argument('--output',type=Path,default=HERE/'final_calibration_20261016_distance120')
    args=parser.parse_args()
    if args.distance_mm<=0:raise ValueError('Distance must be positive')
    summary=[]
    for label, source, title in [
        ('previous', HERE/'final_calibration_20261016_previous', '上一轮ID1基本固定的12组蓝圈样本（约12cm距离约束）'),
        ('latest', HERE/'final_calibration_20261016_174100', '最新一轮12组蓝圈样本（约12cm距离约束）')]:
        folder=args.output/label
        if (folder/'hand_eye.json').exists():
            result=read_json(folder/'hand_eye.json')
            if result.get('distance_measurement',{}).get('fit_radius_mm')!=args.distance_mm:
                raise ValueError('Output contains a different distance assumption; choose a new --output folder')
            summary.append(dict(batch=label,rms_px=result['rms_px_selected'],distance_mm=args.distance_mm,
                                method=result['selected_hand_eye_method']));continue
        folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'inputs').exists():shutil.copytree(source/'inputs',folder/'inputs')
        model,K,dist,points,obs,center,reference=prepare(folder)
        seeds=[];solutions=[]
        existing=read_json(source/'hand_eye.json');mapping=existing['calibrated_joint_mapping']
        seeds.append(dict(camera=np.array(existing['T_gripper_camera']),board=np.array(existing['T_base_board']),
                          delta=np.radians(mapping['angle_reference_deg'])-reference,
                          rates=np.array(mapping['signed_degrees_per_raw']),method='previous_best_fit_seed'))
        outline=points[[0,7,47,40]]
        for signs in itertools.product([-1,1],repeat=3):
            rates=np.array([signs[0],signs[1],signs[2],signs[2],1])*300/1023
            observations=[dict(o,angles=reference+(o['unwrapped'][:5]-center)*np.radians(rates)) for o in obs]
            attempts=[]
            for method in ['PARK','HORAUD','ANDREFF','DANIILIDIS']:
                try:
                    X,Y,spread,_=_closed_form(observations,model,np.zeros(5),outline,HAND_EYE_METHODS[method])
                    if np.isfinite(X).all():attempts.append(dict(camera=X,board=Y,spread=spread,method=method))
                except (ValueError,cv2.error):pass
            if attempts:
                initial=min(attempts,key=lambda x:x['spread'])
                try:fit=_refine_hand_eye(observations,model,K,dist,points,initial['camera'],initial['board'],[1,2,3])
                except (ValueError,cv2.error,np.linalg.LinAlgError):continue
                seeds.append(dict(fit,rates=rates,method=initial['method']+'_'+str(signs)))
        for i,seed in enumerate(seeds):
            fixed=fit_radius(obs,model,K,dist,points,center,reference,seed['rates'],seed,args.distance_mm,False)
            fixed['method']=seed['method']+'_fixed_rates';solutions.append(fixed)
            delta=fixed['reference']-reference
            fitted=fit_radius(obs,model,K,dist,points,center,reference,fixed['rates'],dict(fixed,delta=delta),args.distance_mm,True)
            fitted['method']=seed['method']+'_fitted_rates';solutions.append(fitted)
            print(label,i,'fixed',round(fixed['rms_px'],5),'fitted',round(fitted['rms_px'],5),flush=True)
        best=min(solutions,key=lambda x:x['rms_px'])
        result=result_json(folder,best,args,title)
        write_json(folder/'hand_eye.json',result)
        rows=[dict(method=x['method'],rms_px=x['rms_px'],distance_mm=float(np.linalg.norm(x['camera'][:3,3])),
                   rates=x['rates'].tolist(),T_gripper_camera=x['camera'].tolist(),T_base_board=x['board'].tolist(),
                   angle_reference_deg=np.degrees(x['reference']).tolist()) for x in sorted(solutions,key=lambda x:x['rms_px'])]
        write_json(folder/'candidate_results.json',dict(candidates=rows,assumed_distance_mm=args.distance_mm))
        # Approximate distance sensitivity; not independent validation of the radius.
        sensitivity=[]
        for radius in [args.distance_mm-20,args.distance_mm,args.distance_mm+20]:
            if radius<=0:continue
            fit=best if radius==args.distance_mm else fit_radius(obs,model,K,dist,points,center,reference,best['rates'],
                dict(best,delta=best['reference']-reference),radius,True)
            sensitivity.append(dict(assumed_radius_mm=radius,rms_px=fit['rms_px']))
        write_json(folder/'distance_sensitivity.json',dict(assumed_not_measured_interval_mm=20,candidates=sensitivity))
        summary.append(dict(batch=label,rms_px=best['rms_px'],method=best['method'],distance_mm=args.distance_mm))
        print('BATCH FINAL',summary[-1],flush=True)
    write_json(args.output/'comparison.json',dict(batches=summary,selected=min(summary,key=lambda x:x['rms_px'])['batch'],
        selection='lowest all-corner training RMS at the same user-derived assumed distance; no samples omitted'))


if __name__=='__main__':main()
