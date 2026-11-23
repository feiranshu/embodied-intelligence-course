# 具身智能课程实验：SO-101 机械臂控制、视觉伺服与语言模型控制

本项目记录一套 SO-101 结构机械臂的四阶段课程实验，涵盖六轴位置控制、姿态与语音操作、相机标定与视觉伺服，以及基于语言模型的机器人控制。第 6、7 周复用第 4 周的机械臂控制代码。硬件使用六只 Feetech SCS215 舵机，配合 Waveshare Bus Servo Adapter (A) 的 CH343 串口，波特率为 1,000,000。

## 快速开始

在 Windows PowerShell 中从项目根目录执行。使用 Python 3.12 创建环境，安装基础依赖后，可以先预览六轴位置命令并运行单轴控制测试。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r week3/requirements.txt
.\.venv\Scripts\python.exe -X utf8 week3/homework_control.py move 0.50 0.50 0.50 0.50 0.50 0.50 --dry-run
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s week4/so101_scs215_portable -p test_move_servo_raw.py
```

干运行打印六轴目标，测试使用模拟总线。连接 USB 转接板、舵机总线及外接电源后，运行只读诊断，检查六轴位置、模式、硬件限位和电压。

```powershell
.\.venv\Scripts\python.exe -X utf8 week4/so101_scs215_portable/diagnose.py
```

默认自动识别 CH343。指定端口时传入 `--port COMx`。每次只运行一个占用机械臂串口的程序。实机点动前，先根据实时反馈和当前关节映射选择目标，确认操作者在场、活动范围已清空并能立即切断电源。单轴控制工具默认预览，添加 `--execute` 才下发动作。单轴工具检查舵机硬件限位，实际关节活动范围以本机标定为准。 [单轴工具](week4/so101_scs215_portable/move_servo_raw.py) · [关节配置](week4/so101_scs215_portable/advanced.json)

## 目录导航

| 路径 | 内容 | 从这里开始 |
| --- | --- | --- |
| `week3/` | 六轴编号、状态读取、关节校准、归一化位置控制与同步写入 | [实验说明](week3/README.md) · [控制脚本](week3/homework_control.py) |
| `week4/` | 本机网页控制、预设姿态与动作、URDF 运动学、离线中文语音和黑球跟随实验 | [工程说明](week4/so101_scs215_portable/README.md) · [进阶操作](week4/so101_scs215_portable/进阶使用说明.md) · [技术交接](week4/so101_scs215_portable/HANDOFF.md) |
| `week6_visual_servo/` | 棋盘格相机内参、眼在手上手眼标定、ArUco 视觉伺服与光流物体跟踪 | [复现指南](week6_visual_servo/README.md) · [视觉伺服网页](week6_visual_servo/part3_visual_servo/README.md) |
| `week7_llm_agent/` | 照片姿态模仿、Function Calling、视觉与语音交互。执行层复用第 4 周关节控制和第 6 周视觉代码 | [人工操作指南](week7_llm_agent/人工操作指南.md) · [实验报告](week7_llm_agent/第七周_LLM_Agent_报告.md) |
| `build_week6_submission.py` | 读取第 6 周标定结果、样本和源码，生成课程提交材料 | [打包脚本](build_week6_submission.py) |

## 复现实验

按所需功能安装依赖。摄像头可按设备名或索引选择，机械臂通过独立串口连接。第 7 周的设备检查和离线测试可在没有模型密钥时运行，调用云端模型时按操作指南配置对应的环境变量。

| 任务 | 依赖 | 运行入口 |
| --- | --- | --- |
| 第 4 周串口诊断与单轴测试 | [基础依赖](week4/so101_scs215_portable/requirements.txt) | `diagnose.py`、`move_servo_raw.py` |
| 第 4 周预设姿态与离线语音 | [进阶依赖](week4/so101_scs215_portable/requirements_advanced.txt) | `advanced.py`、`start_voice.bat` |
| 第 6 周手眼标定与 ArUco 视觉伺服 | [视觉依赖](week6_visual_servo/requirements.txt) | `part2_hand_eye_calibration/web_app.py`（端口 8765）、`part3_visual_servo/web_app.py`（端口 8766） |
| 第 7 周姿态模仿、工具调用与全模态交互 | [Agent 依赖](week7_llm_agent/requirements.txt) | `part1_imitation/imitate.py`、`part2_function_calling/agent.py`、`part3_omni_agent/omni_agent.py` |

手眼采集使用固定在末端的相机和固定于工作台的棋盘。视觉伺服使用 ArUco 标记，开始时可在网页中检查识别和步进预览。第 7 周入口默认使用模拟机械臂。照片模仿将模型识别结果匹配到已记录的姿态。工具调用可用 `--read-only` 隐藏运动工具。 [手眼采集与视觉实验](week6_visual_servo/README.md) · [Agent 命令示例](week7_llm_agent/README.md)

第 6 周统一标定网页位于 [本机端口 8765](http://127.0.0.1:8765/)，可用 [启动器](week6_visual_servo/part2_hand_eye_calibration/Start_Keyboard_Capture.cmd) 打开最终数据。

## 参考结果与验证范围

| 实验 | 参考结果 | 记录 |
| --- | --- | --- |
| 六轴通信与单轴位置控制 | 2026-11-22 在 COM8 读通六轴，均为位置模式。ID 4 从 192 移到 201，进入目标 204 的 ±3 格容差。释放后扭矩回读为 0 | [诊断入口](week4/so101_scs215_portable/diagnose.py) |
| 相机内参 | 20 张 640 × 480 棋盘图像，重投影 RMS 为 **0.498 px** | [内参 JSON](week6_visual_servo/part1_camera_calibration/camera_intrinsics.json) |
| 手眼标定 | 20 组修正后的样本，12 组拟合 RMS **3.42 px**，8 组留出 RMS **36.39 px** | [最终结果](week6_visual_servo/part2_hand_eye_calibration/最终标定结果/README.md) |
| 语言模型控制 | 课程报告记录了照片模仿、语音指令和“挥手”“点头”连贯动作的实机演示 | [第 7 周报告](week7_llm_agent/第七周_LLM_Agent_报告.md) |

ID 4 腕部俯仰改装后的已记录原始范围是 `104–758`。当前配置将 `kinematics.calibrated` 设为 `false`，笛卡尔实机控制及依赖该模型的实机追踪需要重新标定和验证。第 6 周的手眼结果须与当时的配套关节映射一起使用。 [当前配置](week4/so101_scs215_portable/advanced.json) · [手眼标定说明](week6_visual_servo/part2_hand_eye_calibration/最终标定结果/README.md)

## 离线检查

安装相应依赖后，从项目根目录分别运行以下测试。第 3 周与第 4 周的基础协议和单轴控制共 22 项通过，第 6 周的视觉伺服与标定可离线复核。第 4 周关节标定测试仍有一项逆运动学范围错误。第 7 周完整测试需要安装 `openai` 等依赖。 [第 4 周关节标定测试](week4/so101_scs215_portable/test_joint_calibration.py) · [第 6 周完整测试命令](week6_visual_servo/准确性核验.md#检查范围)

```powershell
python -X utf8 -m unittest discover -s week3 -p test_offline.py
python -X utf8 -m unittest discover -s week4/so101_scs215_portable -p test_offline.py
python -X utf8 -m unittest discover -s week4/so101_scs215_portable -p test_move_servo_raw.py
python -X utf8 -m unittest discover -s week6_visual_servo -p 'test_*.py'
python -X utf8 -m unittest discover -s week7_llm_agent -p test_offline.py
```

使用快速开始创建的环境时，将上面的 `python` 替换为 `.\.venv\Scripts\python.exe`。离线检查覆盖协议、限位和软件流程，实机动作以舵机反馈和现场记录为准。
