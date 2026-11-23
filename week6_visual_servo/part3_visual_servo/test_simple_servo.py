"""Offline integration checks: real marker detection, calibrated IK and cleanup."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from common import arm_model,read_json,calibrated_gripper_pose,calibration
from part3_visual_servo.visual_servo import MarkerDetector,camera_step
from part3_visual_servo.motion_model import matching_motion_model
from part3_visual_servo.web_app import App

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'part2_hand_eye_calibration/最终标定结果'


class SimpleServoTests(unittest.TestCase):
    @staticmethod
    def synthetic_calibration(model):
        return {'calibrated_joint_mapping': {
            'raw_reference_unwrapped': model.zero.tolist(),
            'angle_reference_deg': (np.degrees(model.reference) + np.arange(1, 6)).tolist(),
            'signed_degrees_per_raw': (np.degrees(model.radians_per_count) * 1.05).tolist(),
        }}

    def test_real_marker_detection_and_depth(self):
        K,dist,size=calibration(ROOT/'part1_camera_calibration/camera_intrinsics.json')
        frame=np.full((size[1],size[0],3),255,np.uint8)
        marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),0,160)
        frame[160:320,240:400]=marker[:,:,None]
        center,depth=MarkerDetector(0,40,'DICT_4X4_50').detect(frame,K,dist)
        np.testing.assert_allclose(center,[319.5,239.5],atol=1)
        self.assertTrue(200<depth<300)
        self.assertIsNone(MarkerDetector(1,40,'DICT_4X4_50').detect(frame,K,dist))

    def test_small_low_contrast_marker(self):
        K,dist,size=calibration(ROOT/'part1_camera_calibration/camera_intrinsics.json')
        frame=np.full((size[1],size[0],3),145,np.uint8)
        marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),0,30)
        marker=np.where(marker>0,145,105).astype(np.uint8)
        frame[225:255,305:335]=marker[:,:,None]
        found=MarkerDetector(0,40,'DICT_4X4_50').detect(frame,K,dist)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found[0],[319.5,239.5],atol=2)

    def test_center_still_available_when_depth_fails(self):
        K,dist,size=calibration(ROOT/'part1_camera_calibration/camera_intrinsics.json')
        frame=np.full((size[1],size[0],3),255,np.uint8)
        marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),0,160)
        frame[160:320,240:400]=marker[:,:,None]
        with patch('cv2.solvePnP',return_value=(False,None,None)):
            center,depth=MarkerDetector(0,40,'DICT_4X4_50').detect(frame,K,dist)
        self.assertIsNone(depth)
        self.assertIsNone(camera_step(center,size,10,None,200,forward=False))

    def test_tilted_marker_center_matches_projection(self):
        K, dist, size = calibration(ROOT/'part1_camera_calibration/camera_intrinsics.json')
        square = np.array([[-20., 20., 0.], [20., 20., 0.], [20., -20., 0.], [-20., -20., 0.]])
        rotation = np.array([0.49, -0.32, 0.15])
        translation = np.array([25., 12., 220.])
        corners, _ = cv2.projectPoints(square, rotation, translation, K, dist)
        expected, _ = cv2.projectPoints(np.zeros((1, 3)), rotation, translation, K, dist)
        detector = MarkerDetector(0, 0, 'DICT_4X4_50')
        with patch.object(detector, '_find', return_value=corners.reshape(4, 2)):
            center, depth = detector.detect(np.zeros((size[1], size[0], 3), np.uint8), K, dist)
        np.testing.assert_allclose(center, expected.reshape(2), atol=0.05)
        self.assertIsNone(depth)

    def test_region_size_changes_completion(self):
        self.assertIsNotNone(camera_step([390,240],(640,480),10,None,200,forward=False))
        self.assertIsNone(camera_step([390,240],(640,480),10,None,200,forward=False,center_region_percent=30))

    def test_calibrated_fk_matches_all_saved_poses(self):
        original=arm_model();result=self.synthetic_calibration(original)
        adapted=matching_motion_model(original,result)
        for sample in read_json(FINAL/'samples.json')['samples']:
            raw=sample['raw']
            actual=adapted.urdf_forward(adapted.angles(raw))[0]
            expected=calibrated_gripper_pose(raw,original,result)[:3,3]
            np.testing.assert_allclose(actual,expected,atol=1e-8)
        self.assertFalse(np.array_equal(original.reference,adapted.reference))

    def test_original_physical_angle_bounds_preserved(self):
        original=arm_model();adapted=matching_motion_model(original,self.synthetic_calibration(original))
        for raw in np.random.default_rng(5).integers(0,1024,(80,6)):
            old=original.angles(raw);new=adapted.angles(raw)
            old_inside=(old>=original.angle_limits[:,0])&(old<=original.angle_limits[:,1])
            new_inside=(new>=adapted.angle_limits[:,0])&(new<=adapted.angle_limits[:,1])
            np.testing.assert_array_equal(old_inside,new_inside)

    def test_inverse_has_matching_forward_model(self):
        original=arm_model();model=matching_motion_model(original,self.synthetic_calibration(original))
        raw=[1000,955,200,220,400,352]
        origin=model.forward(raw)
        solved=model.inverse(origin,raw,allow_crossing=False)
        np.testing.assert_allclose(model.forward(solved),origin,atol=2)
        for delta in ([10,0,0],[-10,0,0],[0,10,0],[0,-10,0],[0,0,10],[0,0,-10]):
            target=model.inverse(origin+delta,raw,allow_crossing=False)
            self.assertLess(np.linalg.norm(model.forward(target)-origin-delta),2)

    def make_app(self,folder,observation):
        app=object.__new__(App)
        app.args=SimpleNamespace(hardware_config=Path('unused'),advanced_config=ROOT.parent/'week4/so101_scs215_portable/advanced.json',port=None,hand_eye=FINAL/'hand_eye.json')
        app.result=read_json(FINAL/'hand_eye.json');app.X=np.array(app.result['T_gripper_camera']);app.source_sha='test'
        app.result.pop('calibration_unwrap_references',None)
        app.result.update(self.synthetic_calibration(arm_model()))
        app.lock=threading.RLock();app.cancel=threading.Event();app.params=dict(marker_id=0,marker_mm=40,step_mm=10,standoff_mm=200,max_steps=30,max_travel_mm=300,center_only=False)
        app.observation=observation;app.frame_time=time.monotonic();app.error='';app.busy=True;app.mode='single';app.status='running';app.message=''
        app.records=[];app.traveled=0;app.size=(640,480);app.run_path=Path(folder)/'run.json';app.arm=None
        return app

    def test_final_offline_mapping_is_not_used_for_motion(self):
        with self.assertRaisesRegex(ValueError, '离线角度展开'):
            matching_motion_model(arm_model(),read_json(FINAL/'hand_eye.json'))

    def exercise_cleanup(self,observation,stop_fails=False,continuous=False,reacquire=False):
        with TemporaryDirectory(dir=ROOT) as folder:
            app=self.make_app(folder,observation);events=[]
            if continuous or reacquire:
                app.mode='auto';app.params.update(max_steps=0,max_travel_mm=0,center_only=True)
            camera_done=threading.Event()
            started=time.monotonic()
            def frame_feed():
                while not camera_done.is_set():
                    if continuous or time.monotonic()-started>=.06:
                        moves=sum(e[0]=='move' for e in events)
                        app.observation=dict(center_px=[50,240] if moves<3 else [320,240],depth_mm=None)
                        app.frame_time=time.monotonic()
                    camera_done.wait(.02)
            camera_worker=threading.Thread(target=frame_feed) if continuous or reacquire else None
            class FakeArm:
                def __init__(self,*_):self.model=arm_model();self.cancel=threading.Event();self.last_move={'projected':False}
                def cartesian_start_pose(self):return None
                def positions(self):return np.array([990,920,300,740,80,276])
                def move_linear(self,delta):
                    events.append(('move',delta))
                def stop(self):
                    events.append(('stop',None))
                    if stop_fails:raise RuntimeError('stop feedback failed')
            bus=SimpleNamespace(close=lambda:events.append(('close',None)))
            with patch('advanced.Arm',FakeArm),patch('servo_bus.load_config',return_value={}),patch('servo_bus.ServoBus',return_value=bus):
                if camera_worker:camera_worker.start()
                try:app.control_loop()
                finally:
                    camera_done.set()
                    if camera_worker:camera_worker.join()
            self.assertFalse(app.busy)
            self.assertEqual([x[0] for x in events][-2:],['stop','close'])
            log=read_json(app.run_path)
            self.assertEqual(log['completed'],log['status']=='target_reached')
            self.assertTrue(log['empirical_mapping_used'])
            return log,events

    def test_single_move_cleanup(self):
        log,events=self.exercise_cleanup(dict(center_px=[50,240],depth_mm=400))
        self.assertEqual(log['status'],'single_complete')
        self.assertEqual(len([e for e in events if e[0]=='move']),1)
        self.assertEqual(len(log['steps']),1)

    def test_target_lost_never_moves_and_closes(self):
        log,events=self.exercise_cleanup(None)
        self.assertEqual(log['status'],'failed')
        self.assertFalse(any(e[0]=='move' for e in events))

    def test_continues_for_multiple_steps_until_centered(self):
        log,events=self.exercise_cleanup(dict(center_px=[50,240],depth_mm=None),continuous=True)
        self.assertEqual(log['status'],'target_reached',log['message'])
        self.assertEqual(len([e for e in events if e[0]=='move']),3)
        self.assertTrue(all(e['requested_camera_mm'][2]==0 for e in log['steps']))

    def test_temporary_loss_waits_then_resumes(self):
        log,events=self.exercise_cleanup(None,reacquire=True)
        self.assertEqual(log['status'],'target_reached',log['message'])
        self.assertEqual(len([e for e in events if e[0]=='move']),3)

    def test_stop_failure_reported_and_port_closed(self):
        log,_=self.exercise_cleanup(dict(center_px=[50,240],depth_mm=400),True)
        self.assertEqual(log['status'],'stop_failed')
        self.assertIn('stop feedback failed',log['message'])


if __name__=='__main__':unittest.main()
