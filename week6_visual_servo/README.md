# 第六周：视觉伺服课堂练习与课后作业

本目录对应[课程网页《EmbodiedAI Week 6: Visual Servoing》“四、课堂练习 & 课后作业”](https://sa87gcdhzg.feishu.cn/wiki/D1jWwPc3PikTAFkHkTkcfgJknEf)的四项任务。代码按 1、2、3、4 分开，公共几何计算放在 `common.py`。所有路径都由脚本位置推导，命令可以在本目录运行；不依赖项目放在桌面上的具体位置。

2026-11-22 复算确认内参结果，手眼矩阵未通过跨批次检查。四项实验的核验方法、数值和验收范围见 [准确性核验](准确性核验.md)。

| 部分 | 子文件夹 | 任务 | 主要结果 |
| --- | --- | --- | --- |
| 1 | `part1_camera_calibration` | 棋盘格相机内参标定，至少 15 张不同视角，方格边长默认 25 mm | `camera_intrinsics.json` |
| 2 | `part2_hand_eye_calibration` | 眼在手上手眼标定，至少 12 个完整棋盘视角 | `hand_eye.json`、`hand_eye.png` |
| 3 | `part3_visual_servo` | ArUco 简单规则视觉伺服 | `run_log.json`（实机时） |
| 4 | `part4_object_tracking` | 自选物体区域＋光流跟踪，令目标居中 | `run_log.json`（实机时） |

## 0. 准备环境和硬件

在 PowerShell 进入本目录：

```powershell
cd '你自己的项目路径\embodied-intelligence-course-main\week6_visual_servo'
conda activate py312
python -c "import cv2, numpy, serial; print(cv2.__version__, numpy.__version__, serial.__version__); print(hasattr(cv2.aruco, 'ArucoDetector'))"
```

本机已找到 `D:\software\anaconda\envs\py312`，其中 Python 3.12、NumPy、OpenCV（含 ArUco）和 pyserial 均已可用；另已为本作业加入 `pygrabber`，用于按 Windows DirectShow 名称选择摄像头。若你把项目复制到别的电脑，先建立/激活自己的 Python 3.12 Conda 环境；上面的检查报缺包时，在该环境运行：

```powershell
python -m pip install -r requirements.txt
```

相机要牢固安装在**末端执行器**上，且采集/控制期间安装位置不能变；否则第二、三、四部分的手眼矩阵无效。棋盘格在第二部分须固定于工作台。机械臂为本项目的 **SCS215** 六关节配置，使用同项目 `so101_scs215_portable` 中已有的 URDF、映射、限位、反馈和运动控制。务必重新确认这些本机配置适合现场机械臂；历史标定值和离线测试不能证明实机运动安全。只允许一个程序占用串口。

**本机连接确认（2026-10-12）：** Windows DirectShow 同时枚举到 `Integrated Camera` 和 `GENERAL WEBCAM`，程序按名称选中的外接摄像头已读取到 640×480 画面。机械臂只读诊断在 CH343 `COM8` 上识别到 6 个 SCS215，均返回位置模式和约 4.9～5.0 V 电压；诊断没有发送运动指令。COM 号可能随 USB 连接变化，运行时请继续使用自动识别或传入当前端口，不要把 `COM8` 当成固定值。

**摄像头不通过串口连接。** 四个程序的 `--camera` 默认值已设为图片中设备名 `GENERAL WEBCAM`：程序通过 Windows DirectShow 枚举名称，找到对应索引后打开该设备；`--port COMx` 才是机械臂串口。若设备实际名称略有不同，可先运行检测工具查看系统报告的名称：

```powershell
python check_camera.py
```

它会打印 DirectShow 设备名并尝试从 `GENERAL WEBCAM` 读取一帧。Windows 11 请到“设置 → 蓝牙和设备 → 摄像头”确认它列在“已连接的摄像头”中并可预览；仅出现在蓝牙配对设备列表还不能确认视频流已提供给桌面应用。然后打开系统“相机”应用确认实时画面，并在“设置 → 隐私和安全性 → 相机”允许桌面应用访问。若名称不同，可将四个程序的 `--camera` 设置为清单中的完整名称，也可传数字索引作手动兼容。若 Windows 未将设备暴露为 DirectShow 摄像头，名称选择会明确报错，不会悄悄切换到其他相机。微软的[摄像头设置说明](https://support.microsoft.com/en-au/windows/hardware/camera/manage-cameras-with-camera-settings-in-windows-11)介绍了连接摄像头的预览与启用方式。

## 1. 相机内参标定

棋盘参数指**内角点**，例如 9×6 内角点对应 10×7 个黑白方格。先量好真实方格边长；默认是课程要求的 25 mm。相机镜头、焦距、拍摄分辨率在四部分中保持不变。

```powershell
python part1_camera_calibration/camera_calibration.py capture --cols 8 --rows 6 --square-mm 25 --count 18
python part1_camera_calibration/camera_calibration.py calibrate --cols 8 --rows 6 --square-mm 25
```

取景窗口里，完整角点可见时按**空格**保存；改变棋盘距离、倾角和画面位置，至少得到 15 张不同视角，按 **Q/Esc** 可提前退出。图像保存在 `part1_camera_calibration/images/`。第二条命令读取图像并写入 `camera_intrinsics.json`，其中有相机矩阵、畸变系数、棋盘规格、每张图像误差及整体重投影 RMS。课程目标约为 1 px；程序会报告真实值，不会把高误差伪装成通过。误差偏大时重拍模糊、几乎重复或边缘角点不清晰的图像，并重新标定。

任务一交接包中的已完成结果也已放入本目录：`part1_camera_calibration/camera_intrinsics.json`，20 张原始照片和两张结果图在 `part1_camera_calibration/evidence/task1_handoff/`。内参是 640×480、8×6 内角点、25 mm 方格，RMS 为 0.498 px；若相机分辨率、镜头或安装方式改变，应重新标定。

## 2. 眼在手上手眼标定

本轮最终收录 20 组原图和六轴反馈。前 12 组拟合重投影 RMS 为 3.42 px，8 组留出重投影 RMS 为 36.39 px。矩阵、配套配置及复核步骤见 [最终标定结果](part2_hand_eye_calibration/最终标定结果/README.md)。

从本目录启动统一标定网页，打开 [标定网页](http://127.0.0.1:8765/)。

```powershell
python part2_hand_eye_calibration/web_app.py
```

也可双击 [Start_Keyboard_Capture.cmd](part2_hand_eye_calibration/Start_Keyboard_Capture.cmd)。默认打开最终样本。另建采集目录时，通过 `--output` 指定路径。只看棋盘可加 `--preview`，手动摆臂采样可加 `--record-only`。

按 **1–6** 选轴，按住 **← / →** 连续调整，松键或失焦后释放驱动。采集时保持相机支架和棋盘固定，完整识别 48 个内角点，并确认六轴读数满足网页提示。前 12 张用于拟合，之后保存的照片用于留出验证。紫色 0 对应棋盘物理原点，启动采集前可用蓝圈确认角点顺序。

一键求解显示矩阵、拟合误差和留出误差。可填写独立测量的棋盘原点坐标来比较实物位置误差。本轮的运动学展开只用于标定复核，视觉伺服实机运动仍有待验证。

## 3. ArUco 简单视觉伺服

**推荐使用网页版。** 启动下方程序，打开 `http://127.0.0.1:8766/`。先退出任务二服务以释放相机和串口。网页提供实时画面、标记中心与深度、可打印的ID0标记、“执行一步”“持续调整直到居中”和“停止”按钮。默认每次请求10 mm，默认只预览，不自动运动。点击持续调整后默认只居中，中心进入可调黄色区域即完成；不要求靠近到200 mm，步数／行程0表示不限。短暂漏检等待重新识别后继续，画面／通信异常仍会停止。

详细步骤、当前标定兼容说明和日志位置见 [任务三网页版使用说明](part3_visual_servo/README.md)。任务三默认读取最终标定文件，识别预览可直接使用。该结果的离线角度展开还需与实时运动学验证。

PowerShell启动：

```powershell
conda activate py312
python .\part3_visual_servo\web_app.py
```

以下为仍保留的命令行预览／运行方式：

使用 `DICT_4X4_50`、ID 0 的 ArUco 标记作默认示例。标记必须固定，`--marker-mm` 填**实测黑色编码正方形边长**（mm），不是纸张边长。先只预览识别：

```powershell
python part3_visual_servo/visual_servo.py --marker-id 0 --marker-mm 40
```

预览窗口按 **Q/空格/Esc** 退出，**不打开串口**。画面中间 40%～60% 的矩形是目标区。实机模式下，标记中心落在画面左/右/上/下 40% 时，程序在对应相机坐标方向请求一次 10 mm 移动；中心进入目标区后，只有估计距离大于预设停止距离＋10 mm 时才向前请求 10 mm，否则结束。一次只处理一个方向，连续稳定看见三帧才发起下一步，丢失目标则停止。

先核实方向与空间边界，再明确启动实机模式：

```powershell
python part3_visual_servo/visual_servo.py --marker-id 0 --marker-mm 40 --standoff-mm 200 --execute
```

程序将要求输入 `RUN`。相机步进会通过第二部分的 `T_gripper_camera` 和当前真实关节反馈换算到基座坐标，再调用现有机械臂直线运动/限位/反馈逻辑。默认最多 30 步、总请求行程 60 mm；请按现场空间进一步缩小 `--max-steps`、`--max-travel-mm` 或 `--step-mm`。标记边长不准确会影响距离估计，所以请留足停止距离。退出后的日志含每一步请求与反馈，`completed` 只在目标居中或达到停止距离时为真。

## 4. 任意纹理物体跟踪（加分题）

这里选用课程允许的“任意物体”方向：人工框选目标区域，OpenCV 角点＋金字塔 Lucas–Kanade 光流跟踪，无须物体类别模型。适合有纹理、轮廓清楚、运动较慢的物体；过于光滑、强反光或被遮挡的目标会丢失。先预览：

```powershell
python part4_object_tracking/object_tracking.py
```

在首帧用鼠标框选目标并按 **Enter**。接下来预览框的位置；按 **Q/空格/Esc** 退出。确认跟踪稳定、第二部分手眼矩阵仍适用后，才能实机运行：

```powershell
python part4_object_tracking/object_tracking.py --execute
```

仍须框选并输入 `RUN`。跟踪器只输出物体中心，没有可靠的绝对深度，因此本部分**只做居中，不做向前靠近**；每步最多 5 mm，默认最多 6 步和 30 mm 总请求行程。跟踪失败、目标离开画面或连续丢失时停止。对于快速移动物体，操作者应先停止运动并重新框选。

## 离线自检与提交

```powershell
python -m unittest discover -s . -p test_offline.py -v
python -m compileall -q common.py part1_camera_calibration part2_hand_eye_calibration part3_visual_servo part4_object_tracking
```

上述检查只验证代码、坐标约定及合成图像下的跟踪；它们**不等于**相机/机械臂现场验收。提交时把真实 `images/`、`camera_intrinsics.json`、`samples/`、`hand_eye.json`，以及第三、四部分的预览视频/实机日志按实际完成情况附上。交接包中的任务一照片和结果已随项目保存；任务二的姿态和棋盘图像需要在本机现场采集。
