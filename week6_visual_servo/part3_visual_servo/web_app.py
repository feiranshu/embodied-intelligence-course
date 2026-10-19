"""Local simple ArUco visual-servo webpage; opening it never starts motion."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from camera_utils import DEFAULT_CAMERA,open_camera
from common import (ARM_CODE,read_json,write_json,calibration,ensure_image_size,
                    load_hand_eye,base_gripper_pose)
from part3_visual_servo.visual_servo import MarkerDetector,camera_step
from part3_visual_servo.motion_model import matching_motion_model


class App:
    def __init__(self,args):
        self.args=args;self.lock=threading.RLock();self.capture_lock=threading.Lock()
        self.done=threading.Event();self.cancel=threading.Event()
        self.camera=None;self.arm=None;self.worker=None;self.camera_thread=None
        self.jpeg=None;self.observation=None;self.frame_time=0.;self.error='';self.last_client_seen=time.monotonic()
        self.camera_name='';self.busy=False;self.message='等待摄像头连接';self.mode='idle'
        self.records=[];self.status='idle';self.traveled=0.;self.run_index=0;self.run_path=None
        self.params=dict(marker_id=0,marker_mm=40.,step_mm=10.,standoff_mm=200.,
                         max_steps=0,max_travel_mm=0.,center_only=True,center_region_percent=20.)
        self.K,self.dist,self.size=calibration(args.intrinsics)
        self.result=read_json(args.hand_eye);self.X=load_hand_eye(args.hand_eye)
        self.source_sha=hashlib.sha256(args.hand_eye.read_bytes()).hexdigest()
        self.detector=MarkerDetector(0,40.,'DICT_4X4_50')
        self.connect_camera()
        self.camera_thread=threading.Thread(target=self.camera_loop,daemon=True);self.camera_thread.start()

    def connect_camera(self):
        with self.capture_lock:
            if self.camera is not None:self.camera.release();self.camera=None
            try:
                self.camera,_,_,self.camera_name=open_camera(self.args.camera,self.size)
                with self.lock:self.error='';self.message='摄像头已连接，等待ID 0标记'
            except Exception as exc:
                with self.lock:self.error=str(exc);self.message='摄像头连接失败，可点击重新连接'

    def camera_loop(self):
        missing=0
        while not self.done.is_set():
            try:
                with self.capture_lock:
                    if self.camera is None:self.done.wait(.1);continue
                    ok,frame=self.camera.read()
                if not ok:raise RuntimeError('摄像头没有返回画面，请关闭其他占用相机的程序并重新连接')
                ensure_image_size(frame,self.size)
                with self.lock:detector=self.detector;region=self.params['center_region_percent']
                detected=detector.detect(frame,self.K,self.dist)
                observation=None if detected is None else dict(center_px=detected[0].tolist(),depth_mm=detected[1])
                missing=missing+1 if observation is None else 0
                w,h=self.size
                low=.5-region/200;high=.5+region/200
                cv2.rectangle(frame,(int(w*low),int(h*low)),(int(w*high),int(h*high)),(255,255,0),2)
                label='TARGET LOST - WAITING' if observation is None else f"ID {detector.marker_id}"+(f"  Z={detected[1]:.0f}mm" if detected[1] is not None else '  CENTER ONLY')
                cv2.putText(frame,label,
                            (12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255) if observation is None else (0,220,0),2)
                ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                with self.lock:
                    self.observation=observation;self.frame_time=time.monotonic();self.error=''
                    if ok:self.jpeg=encoded.tobytes()
                    active=self.busy;client_age=time.monotonic()-self.last_client_seen
                # Brief detection misses do not terminate the run; no new command is
                # issued until a newly decoded observation is available.
                if active and client_age>3:self.request_stop('网页连接中断，已请求停止')
            except Exception as exc:
                with self.lock:self.error=str(exc);self.observation=None
                self.request_stop('摄像头异常，已请求停止')
                self.done.wait(.15)

    def state(self):
        with self.lock:
            o=self.observation
            return dict(ok=True,app='week6_part3',busy=self.busy,mode=self.mode,status=self.status,
                message=self.message,error=self.error,camera_name=self.camera_name,params=self.params,
                observation=o,steps=len(self.records),traveled_mm=self.traveled,
                records=self.records[-20:],result_source=self.result.get('calibration_dataset',{}).get('label','手眼结果'),
                hand_eye_rms_px=self.result.get('rms_px_selected'),log_file=str(self.run_path) if self.run_path else None)

    def configure(self,data):
        p={k:data.get(k,v) for k,v in self.params.items()}
        for key in ['marker_mm','step_mm','standoff_mm','center_region_percent']:
            p[key]=float(p[key])
            if not math.isfinite(p[key]) or p[key]<=0:raise ValueError(f'{key}必须是有限正数')
        p['marker_id']=int(p['marker_id']);p['max_steps']=int(p['max_steps']);p['center_only']=bool(p['center_only'])
        p['max_travel_mm']=float(p['max_travel_mm'])
        if not 0<=p['marker_id']<50 or p['max_steps']<0:raise ValueError('标记ID需为0～49，次数0表示不限，不能为负数')
        if not math.isfinite(p['max_travel_mm']) or p['max_travel_mm']<0:raise ValueError('总行程0表示不限，不能为负数')
        if p['center_region_percent']>100:raise ValueError('中央区域比例不能超过100%')
        detector=MarkerDetector(p['marker_id'],p['marker_mm'],'DICT_4X4_50')
        with self.lock:
            if self.busy:raise RuntimeError('请先停止当前动作，再修改参数')
            self.params=p;self.detector=detector;self.observation=None
            self.message='参数已更新，等待新的识别画面'

    def request_stop(self,message='已请求停止；正在等待驱动停止反馈'):
        self.cancel.set()
        with self.lock:
            if self.arm is not None:self.arm.cancel.set()
            if self.busy:self.message=message

    def start(self,mode):
        if mode not in ('single','auto'):raise ValueError('未知模式')
        with self.lock:
            if self.busy:raise RuntimeError('已有动作正在执行')
            if self.error:raise RuntimeError(self.error)
            self.busy=True;self.mode=mode;self.status='running';self.message='连接机械臂并检查当前姿态'
            self.cancel.clear();self.last_client_seen=time.monotonic();self.records=[];self.traveled=0.;self.run_index+=1
            self.run_path=HERE/'runs'/f'{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()}.json'
            self.worker=threading.Thread(target=self.control_loop,daemon=True);self.worker.start()

    def save(self):
        with self.lock:
            write_json(self.run_path,dict(status=self.status,completed=self.status=='target_reached',
                steps=self.records,requested_travel_mm=self.traveled,params=self.params,
                hand_eye_file=str(self.args.hand_eye),hand_eye_sha256=self.source_sha,
                empirical_mapping_used=bool(self.result.get('calibrated_joint_mapping')),
                physical_distance_independently_measured=False,message=self.message))

    def control_loop(self):
        bus=arm=None;stop_errors=[]
        try:
            from advanced import Arm
            from servo_bus import ServoBus,load_config
            bus=ServoBus(load_config(self.args.hardware_config),self.args.port)
            arm=Arm(read_json(self.args.advanced_config),bus)
            arm.model=matching_motion_model(arm.model,self.result)
            with self.lock:self.arm=arm
            if self.cancel.is_set():raise InterruptedError('操作者停止')
            arm.cartesian_start_pose()
            after=0.
            while not self.cancel.is_set():
                with self.lock:
                    obs=self.observation;stamp=self.frame_time;p=dict(self.params);error=self.error
                if error:raise RuntimeError(error)
                if time.monotonic()-stamp>2:raise RuntimeError('实时图像已停止更新')
                if obs is None:
                    with self.lock:self.status='waiting_target';self.message='暂未识别目标：暂停发出新运动，重新识别后自动继续'
                    self.cancel.wait(.04);continue
                if stamp<=after:self.cancel.wait(.03);continue
                if not p['center_only'] and obs['depth_mm'] is None:
                    with self.lock:self.status='waiting_depth';self.message='标记已识别，等待有效距离；可选择只居中'
                    self.cancel.wait(.04);continue
                with self.lock:self.status='running'
                delta=camera_step(obs['center_px'],self.size,p['step_mm'],obs['depth_mm'],p['standoff_mm'],forward=not p['center_only'],
                                  center_region_percent=p.get('center_region_percent',20.))
                if delta is None:
                    with self.lock:self.status='target_reached';self.message='目标已居中'+('' if p['center_only'] else '，已到停止距离或不足一个步长')
                    break
                if (p['max_steps']>0 and len(self.records)>=p['max_steps']) or (p['max_travel_mm']>0 and self.traveled+np.linalg.norm(delta)>p['max_travel_mm']+1e-6):
                    with self.lock:self.status='run_limit';self.message='达到本次设定的步数或总行程，尚未确认目标到位'
                    break
                raw=arm.positions();T_bg=base_gripper_pose(raw,arm.model)
                delta_base=(T_bg@self.X)[:3,:3]@delta
                with self.lock:self.message=f'执行第{len(self.records)+1}步：相机方向请求 {delta.tolist()} mm'
                arm.move_linear(delta_base)
                record=dict(center_px=obs['center_px'],depth_mm=obs['depth_mm'],raw_before=raw.tolist(),
                    requested_camera_mm=delta.tolist(),requested_base_mm=delta_base.tolist(),
                    feedback=arm.last_move or {})
                with self.lock:self.records.append(record);self.traveled+=float(np.linalg.norm(delta))
                self.save();after=time.monotonic()
                if record['feedback'].get('projected'):
                    with self.lock:self.status='partial_move';self.message='本步只完成部分可行位移，已结束本次运行'
                    break
                if self.mode=='single':
                    with self.lock:self.status='single_complete';self.message='单步完成，请观察画面后继续'
                    break
                self.cancel.wait(.15)
            if self.cancel.is_set():
                with self.lock:self.status='operator_or_vision_stop';self.message='已收到停止请求，正在停止驱动'
        except Exception as exc:
            with self.lock:self.status='failed';self.message=f'{type(exc).__name__}: {exc}'
        finally:
            if arm is not None:
                try:arm.stop()
                except Exception as exc:stop_errors.append(str(exc))
            if bus is not None:
                try:bus.close()
                except Exception as exc:stop_errors.append(str(exc))
            with self.lock:
                if stop_errors:self.status='stop_failed';self.message+='；停止/关闭失败：'+'；'.join(stop_errors)
                self.arm=None;self.busy=False;self.mode='idle'
            self.save()

    def close(self):
        self.request_stop();self.done.set()
        if self.worker:self.worker.join(timeout=8)
        if self.camera_thread:self.camera_thread.join(timeout=2)
        with self.capture_lock:
            if self.camera is not None:self.camera.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def send(self,data,kind='application/json; charset=utf-8',status=200):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path=='/':self.send((HERE/'web_ui.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/api/state':
                with self.server.app.lock:self.server.app.last_client_seen=time.monotonic()
                self.send(self.server.app.state())
            elif self.path.startswith('/frame.jpg'):
                with self.server.app.lock:data=self.server.app.jpeg
                self.send(data or b'', 'image/jpeg',200 if data else 503)
            elif self.path=='/marker.svg':self.send((HERE/'marker_id0_40mm.svg').read_bytes(),'image/svg+xml')
            elif self.path=='/print-marker':self.send((HERE/'print_marker.html').read_bytes(),'text/html; charset=utf-8')
            else:self.send(dict(error='not found'),status=404)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_POST(self):
        # Only the local page may initiate commands; reject cross-site form submissions.
        if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
            self.send(dict(ok=False,error='请从本机任务三页面操作'),status=403);return
        if 'application/json' not in self.headers.get('Content-Type',''):
            self.send(dict(ok=False,error='JSON required'),status=400);return
        try:
            data=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length','0')),16384)) or b'{}')
            app=self.server.app
            if self.path=='/api/config':app.configure(data)
            elif self.path=='/api/start':app.start(data.get('mode','single'))
            elif self.path=='/api/stop':app.request_stop()
            elif self.path=='/api/reconnect':
                if app.busy:raise RuntimeError('请先停止动作')
                app.connect_camera()
            else:raise ValueError('未知请求')
            self.send(app.state())
        except Exception as exc:self.send(dict(ok=False,error=str(exc)),status=400)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',default=DEFAULT_CAMERA)
    parser.add_argument('--port')
    parser.add_argument('--http-port',type=int,default=8766)
    parser.add_argument('--intrinsics',type=Path,default=HERE.parent/'part1_camera_calibration/camera_intrinsics.json')
    parser.add_argument('--hand-eye',type=Path,default=HERE.parent/'part2_hand_eye_calibration/最终标定结果/hand_eye.json')
    parser.add_argument('--advanced-config',type=Path,default=ARM_CODE/'advanced.json')
    parser.add_argument('--hardware-config',type=Path,default=ARM_CODE/'config.json')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args();app=None;server=ThreadingHTTPServer(('127.0.0.1',args.http_port),Handler)
    try:
        app=App(args);server.app=app
        url=f'http://127.0.0.1:{args.http_port}/';print('Task 3:',url,flush=True)
        if not args.no_browser:webbrowser.open(url)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:print('正在关闭任务三服务',flush=True)
    finally:
        if app:app.close()
        server.server_close()


if __name__=='__main__':main()
