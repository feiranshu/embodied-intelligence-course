# 第六周：视觉伺服课堂练习与课后作业

本目录对应[课程网页《EmbodiedAI Week 6: Visual Servoing》“四、课堂练习 & 课后作业”](https://sa87gcdhzg.feishu.cn/wiki/D1jWwPc3PikTAFkHkTkcfgJknEf)的四项任务。代码按 1、2、3、4 分开，公共几何计算放在 `common.py`。所有路径都由脚本位置推导，命令可以在本目录运行；不依赖项目放在桌面上的具体位置。

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

## 2. 眼在手上手眼标定网页

### 2026-10-16 最终结果

你已确认约12 cm的测量不可靠，因此撤销120 mm约束。当前最终选用上一轮ID1基本固定的12组蓝圈样本，在已测试的无距离约束候选中，全部角点训练重投影RMS最低，为 **8.8364 px**。另一轮无距离约束最低测试RMS为25.9751 px。输出的376.13 mm是未实测验证的拟合距离。两轮独立求解，没有混合数据。相机与末端安装关系未变，移动底座本身不使手眼矩阵失效。被撤销的120 mm版本保存在 `part2_hand_eye_calibration/最终标定结果_已撤销12cm约束/`。

矩阵、配套FK映射、完整样本快照、局限和PowerShell复核方法见 [最终结果使用说明](part2_hand_eye_calibration/最终标定结果/README.md)。主目录结果已同步。**本结果必须配套JSON中的经验角度映射；8.84 px是训练拟合误差，关节1、2方向与安装距离仍未被独立验证，尚不能认定实机定位准确。** 第三部分已配套同一经验FK／IK，第四部分仍使用原控制模型。下文网页求解流程会重新使用当前采集列表，并可能覆盖主目录矩阵；独立的最终结果包不受影响。

把棋盘固定在桌面、摄像头固定在机械臂末端，然后从本目录启动本机网页：

```powershell
python part2_hand_eye_calibration/web_app.py
```

程序会按 `GENERAL WEBCAM` 设备名打开相机，自动识别舵机串口，并在浏览器打开 `http://127.0.0.1:8765/`。若自动识别不到串口，可显式指定当前端口，例如：

请通过上面的 Python 命令启动服务，再打开 `http://127.0.0.1:8765/`。不要双击 `web_ui.html`；直接打开 HTML 文件只会显示静态页面，无法连接摄像头、机械臂或姿态采集接口。

```powershell
python part2_hand_eye_calibration/web_app.py --servo-port COM8
```

网页包括以下操作：

1. **实时画面**：叠加显示 8×6 棋盘角点和稳定状态；确认整块棋盘完整可见。
2. **六轴连续控制**：每个舵机有 `− / +` 按键，步长可选 4、8、16 ticks。鼠标或触屏按住按键会连续调整，松开、网页失焦或切换标签页时停止。键盘按 `1–6` 选择舵机，再按住 `← / →` 控制正向 / 反向运动；松开方向键停止。页面的“停止当前动作”按钮也可中止运动。手动微调不受 `advanced.json` 映射工作区限制；舵机位置模式本身的目标范围仍生效。每步只调整一个舵机；请留意机械臂周围空间并准备随时停止。
3. **采集当前姿态**：只要求检测到完整棋盘，且关节1～5的反馈都不在 **0～30、1000～1022（含边界）** 内。命中区间的关节ID、名称、当前读数会列在网页最上方；关节6不参与该区间检查。点击采集时后端会用新读取的位置再次检查，符合条件才保存照片和六轴原始反馈到 `part2_hand_eye_calibration/samples/`。不要求连续稳定帧、姿态不同、位于软件映射工作区或停止微调，也不检查其他读数区间。这项规则只影响采集，不禁止控制关节转动。可分多次运行，已有样本会继续显示和追加。更新程序后需重启网页服务并刷新页面。

   顶部还会显示通信失败、等待反馈或棋盘未检测等不能采集的实际原因。若显示“无回复”，表示尚未取得位置，不能判断关节是否命中区间；先检查舵机外接电源和总线连接。USB串口显示正常不等于舵机已有供电和反馈。Python代码更新后，在原服务窗口按Ctrl+C结束，再双击启动CMD；已有服务运行时，CMD只会打开原页面，不会替换运行中的代码。

   **删除记录：** 每条采集记录右侧有“删除”按钮，也可点击“删除全部采集记录”。列表、样本数量和求解按钮会同步更新。被删除的记录及照片移入 `samples/deleted_records/` 归档，不再参与求解；旧矩阵及结果图也会归档，需使用剩余样本重新求解。采集或求解进行中不能同时删除。
4. **生成手眼矩阵与误差**：至少记录 12 张后，一键求解。页面展示 `T_gripper_camera`、重投影 RMS、棋盘位置散布、原始 FK 与偏差修正后的 RMS，以及估计的 ID 2–4 关节角度偏差。结果写入 `part2_hand_eye_calibration/hand_eye.json` 和 `hand_eye.png`。

离线求解已取消接缝区和运动限位筛选，使用清单内图片及有效的六轴原始反馈进行角度换算。输出中的 `camera_origin_distance_to_gripper_mm` 是相机光学原点至 URDF 末端 `gripper_frame_link` 原点的拟合直线距离。它的可信程度取决于图像和关节姿态的一致性；本次 pose_002 至 pose_013 的逐图误差和数值结果见 `part2_hand_eye_calibration/calibration_report.md`。

**蓝圈固定角点方向：** 保留棋盘纸右下角、棋盘外侧的蓝色圆圈。网页自动识别蓝圈，将最近的那个物理内角点固定为紫色“0”，同时在蓝圈上画绿色圆圈和连线。画面旋转或检测器反转角点顺序时，会重新排序，不再以画面左上角作为起点。页面显示“蓝圈已识别”才表示获得了蓝圈的绝对方向参考；蓝圈短暂识别不到时沿用相邻画面的方向并提示，首次未识别时明确提示方向未确认，不增加新的禁采条件。为获得可靠定向，请让蓝圈完整可见。

网页采集会把定向后的48个角点写入记录的 `ordered_corners`，求解直接使用它们，避免重新检测时再次翻转。旧样本没有保存顺序时，求解会尝试从照片里的蓝圈重新定向；没有蓝圈的旧照片无法自动保证物理原点一致，不能假定本次修改已经修复旧矩阵。已有记录不会自动删除或覆盖。

原点 XYZ 输入框可选，单位 mm，坐标系相对机械臂基座；若填写，应测量现在蓝圈旁“0”角点的实际坐标。未填写时的位置和方向散布只衡量样本一致性，不代表真实位置误差。

程序只绑定本机地址 `127.0.0.1`。完成后在启动它的 PowerShell 按 **Ctrl+C** 退出。关掉网页标签不会结束采集服务。

**姿态采集建议：** 12 张是最低数量。程序允许保存重复姿态，但求解需要足够的关节转角和位置变化；视角变化不足时，求解可能失败或结果不稳定。机械臂按钮和键盘方向键会真实驱动舵机：按住期间持续运动，松开即请求停止。请根据实机情况控制步长和运动时间。

求解器会尝试估计肩部、肘部和腕部俯仰的常量角度偏差，并将运动学修正建议写在 `hand_eye.json` 的 `kinematics_patch_suggestion` 字段；不会自动改写机械臂配置。先检查矩阵、重投影 RMS、位置散布和偏差建议，再决定是否应用运动学修正。第三、四部分的实机控制仍依赖 `advanced.json` 中当前的运动学配置；在检查并验证建议前，不要把本次结果直接用于实机视觉运动。

如需在不使用网页的情况下只读记录，也可运行 `python part2_hand_eye_calibration/record_poses.py`；检测到完整棋盘后按空格记录、按 Q 退出。命令行采集同样不检查映射工作区、接缝或重复姿态。之后可运行 `python part2_hand_eye_calibration/hand_eye.py solve` 从命令行求解。

## 3. ArUco 简单视觉伺服

**推荐使用网页版：** 双击最外层项目根目录 `启动任务三视觉伺服.cmd`，打开 `http://127.0.0.1:8766/`。先退出任务二服务以释放相机和串口。网页提供实时画面、标记中心与深度、可打印的ID0标记、“执行一步”“持续调整直到居中”和“停止”按钮。默认每次请求10 mm，默认只预览，不自动运动。点击持续调整后默认只居中，中心进入可调黄色区域即完成；不要求靠近到200 mm，步数／行程0表示不限。短暂漏检等待重新识别后继续，画面／通信异常仍会停止。

详细步骤、当前标定兼容说明和日志位置见 [任务三网页版使用说明](part3_visual_servo/README.md)。本次已在任务三运行时统一手眼结果和经验FK／IK，保留原运动范围；未改写week4配置。离线检查通过不代表已完成实机闭环验收。

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
