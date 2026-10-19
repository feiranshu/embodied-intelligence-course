"""Fit rotation-invariant diagnostics offline; do not write robot calibration."""
from pathlib import Path
import sys
import copy
import itertools
import math
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import read_json,write_json,board_points,arm_model
from part2_hand_eye_calibration.hand_eye import _closed_form,_refine_hand_eye,HAND_EYE_METHODS


def main():
    folder=HERE/'stationary_id1_audit_20261016'
    audit=read_json(folder/'observations.json')
    rows=audit['observations'];raw=np.array([r['raw'] for r in rows],float)
    pairs=audit['pairs']; delta=np.array([p['raw_delta'] for p in pairs],float)
    measured=np.array([p['camera_deg'] for p in pairs])
    # With fixed base, the three pitch axes are parallel and wrist-roll is perpendicular.
    # Relative rotation angle depends on pitch sum and roll change, independent of zeros/X.
    def residual(v):
        pitch=np.radians(delta[:,1]*(-300/1023)+delta[:,2]*v[0]+delta[:,3]*v[1])
        roll=np.radians(delta[:,4]*v[2])
        prediction=np.degrees(2*np.arccos(np.clip(np.abs(np.cos(pitch/2)*np.cos(roll/2)),0,1)))
        return prediction-measured
    fits=[]
    for seed in itertools.product([-.4,.4],[-.4,.4],[.3,.4]):
        p=np.array(seed,float);damping=.001;e=residual(p);cost=e@e
        for _ in range(100):
            J=np.column_stack([(residual(p+np.eye(3)[i]*1e-5)-e)/1e-5 for i in range(3)])
            step=-np.linalg.solve(J.T@J+damping*np.eye(3),J.T@e)
            trial=residual(p+step);newcost=trial@trial
            if newcost<cost:
                p+=step;e=trial
                if cost-newcost<1e-10:break
                cost=newcost;damping=max(damping/3,1e-10)
            else:damping*=5
            if damping>1e12:break
        fits.append(dict(signed_degrees_per_count=p.tolist(),rotation_pair_rms_deg=float(np.sqrt(np.mean(e**2)))))
    fits.sort(key=lambda r:r['rotation_pair_rms_deg'])
    baseline=residual(np.array([-300/1023,300/1023,300/1023]))
    output=dict(baseline_pair_rms_deg=float(np.sqrt(np.mean(baseline**2))),fits=fits,
        note='Diagnostic only. Fixed-base approximation ignores 1 count base drift. Global pitch-sign ambiguity remains.')
    write_json(folder/'rotation_mapping_fit.json',output)
    print('INVARIANT_FIT',output,flush=True)
    # Evaluate both pitch-sign branches with the full URDF and reprojection optimizer.
    model=arm_model();intr=read_json(HERE.parent/'part1_camera_calibration/camera_intrinsics.json')
    K=np.array(intr['camera_matrix']);dist=np.array(intr['dist_coeffs']);pts=board_points(8,6,25)
    manifest=read_json(folder/'samples_snapshot.json')['samples']
    outline=pts[[0,7,47,40]]
    best=fits[0]['signed_degrees_per_count']
    full=[]
    for branch in [1,-1]:
        pitch3,pitch4,roll=best[0]*branch,best[1]*branch,abs(best[2])
        observations=[]
        for row,record in zip(rows,manifest):
            q=model.angles(row['raw'])
            unfolded=model.unwrap(row['raw'])
            q[2]=(unfolded[2]-model.zero[2])*math.radians(pitch3)+model.reference[2]
            q[3]=(unfolded[3]-model.zero[3])*math.radians(pitch4)+model.reference[3]
            q[4]=(unfolded[4]-model.zero[4])*math.radians(roll)+model.reference[4]
            observations.append(dict(name=row['name'],angles=q,corners=np.array(record['ordered_corners'],np.float32).reshape(-1,1,2),T_cm=np.array(row['Tct'])))
        candidates=[]
        for method_name,method in HAND_EYE_METHODS.items():
            try:
                X,Y,_,_=_closed_form(observations,model,np.zeros(5),outline,method)
                nominal=_refine_hand_eye(observations,model,K,dist,pts,X,Y,[])
                adjusted=_refine_hand_eye(observations,model,K,dist,pts,nominal['camera'],nominal['board'],[1,2,3])
                picked=adjusted if adjusted['rms_px']<nominal['rms_px'] else nominal
                summary=dict(method=method_name,rms_px=picked['rms_px'],nominal_rms_px=nominal['rms_px'],
                    distance_mm=float(np.linalg.norm(picked['camera'][:3,3])),offsets_deg=np.degrees(picked['delta']).tolist(),
                    T_gripper_camera=picked['camera'].tolist())
                candidates.append(summary)
                print('BRANCH',branch,'RATES',pitch3,pitch4,roll,summary,flush=True)
            except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:
                print('SKIP',branch,method_name,str(exc),flush=True)
        full.append(dict(branch=branch,signed_degrees_per_count=[pitch3,pitch4,roll],candidates=candidates))
        write_json(folder/'mapping_sensitivity_results.json',{'rotation_fit':output,'branches':full})


if __name__=='__main__':main()
