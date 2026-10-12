# 课程项目交接说明

整理日期：2026-10-12。

## 交付范围

本次整理自 `C:\Users\kanzaler\Desktop\具身智能` 的现有成果，进阶工程最后修改时间为 2026-10-02。原有 `so101_scs215_position_homework/` 保留，后续工程放在 `so101_scs215_portable/`，两者各自使用自己的配置与协议实现。

| 位置 | 用途 |
|---|---|
| `so101_scs215_portable/` | 后续日常调试入口，含网页控制、姿态/动作、IK、语音及视觉实验 |
| `so101_scs215_position_homework/` | 2026-09-26 的独立六轴归一化位置控制作业 |
| `classroom_servo.py`、`课堂练习报告.md` | 双舵机编号、90°/45°限位、0～1 同步位置控制及原报告 |
| `read_servo.py`、`write_servo.py`、`set_id.py` | 原课堂协议样例，串口仍为占位符；日常控制使用完整工程入口 |
| `Hackathon任务报告.md`、`演示视频/` | 四项任务的报告及四段原有视频 |
| `林邵坪24307090026.md` | 基于当时本地 LeRobot 版本的功能梳理，属于文档核对，不是运行验收 |
| `so101_scs215_portable/verification/` | 保留的关键实机记录、历史检查摘要和四段合成语音测试录音 |

未重复提交旧压缩包、模型下载 ZIP、完整 LeRobot 第三方仓库、虚拟环境、缓存和调试备份。原文件仍保留在上述本地目录。控制源码与配置按现有版本复制，本次没有修改运动逻辑或标定数值。

## 完成状态与证据

| 任务 | 已有成果 | 验收边界 |
|---|---|---|
| 基础位置作业 | 六轴编号、校准、归一化同步运动、单轴测试和干运行 | 当前方案的线性校准区间不能跨反馈端点；现场状态须重新读取 |
| 双舵机课堂实验 | 脚本、原理报告和反馈检查 | 2026-09-21 原报告仍记录现场验收待完成，不代表当前硬件仍处于当日故障状态 |
| 1.1 预设姿态与动作 | 姿态记录、键盘切换、动作序列及时间/速度设置；三段姿态视频 | 历史记录已有姿态实机确认；“动作2/3”视频是编号姿态，不能据文件名当作组合动作验收 |
| 1.2 IK 与六方向定距运动 | URDF、关节映射、限位 IK、分段运动、不可达反馈 | 旧说明按此前确认登记为完成；现有材料缺完整六方向与定距验收、独立演示及外部距离测量，不据模型位移认定精度通过 |
| 2.1 语音控制 | Vosk 中文转录、固定口令、停止处理、可选语言模型；语音实机视频 | 历史记录确认解决；合成录音测试不等于新电脑的麦克风或现场运动验收 |
| 2.2 视觉闭环 | 黑球检测、实验性跟随、网页控制、四点指向标定流程 | **未完成**。摄像头固定、空间对应关系及整屏跟随尚未完成验收；代码存在不等于闭环效果通过 |

原始材料见 [Hackathon 报告](Hackathon任务报告.md)、[进阶使用说明](so101_scs215_portable/进阶使用说明.md) 和 [工程技术交接](so101_scs215_portable/HANDOFF.md)。遇到“已完成”标记，应同时核对其验收范围和实机记录。

## 安装与接手顺序

硬件为六只 Feetech SCS215，总线曾报告型号 `8965`；转接板为 Waveshare Bus Servo Adapter (A)，CH343，波特率 `1,000,000`。不能直接套用 STS3215 的寄存器和校准。ID 1～6 依次为底座、肩部、肘部、腕部俯仰、腕部旋转、夹爪。

1. 安装标准 Windows Python 3.10 或更新版本；进入 `so101_scs215_portable/`，运行 `setup_advanced_windows.bat` 安装进阶依赖。
2. 先用无硬件演示检查软件环境；本次复查所用环境为 CPython 3.11.15。

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 advanced.py --simulate text "立正"
   .\start_advanced_demo.bat
   ```

3. 接好 USB、总线及舵机外部供电后，先运行只读检查：

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 diagnose.py
   ```

   核对六个 ID、型号、模式、位置和供电。默认串口为 `auto`，不要照抄旧记录中的 COM3/COM8；需要时通过配置或 `--port` 指定实际端口。

4. 操作者在场、路径清空并能立即切断外接电源后，再使用实机入口：`start_cartesian.bat` 为姿态/末端控制网页，`start_voice.bat` 为终端语音控制。`start_vision.bat` 是未验收的视觉实验入口，不能作为已完成任务直接验收。

一次只运行一个占用控制串口的程序。换装、挪动机械臂或更换舵机后重新核对运动范围与标定；配置中的 `motion_calibrated`、`kinematics.calibrated` 是原装配的保存状态，不是新现场的验证结果。两个工程的 `config.json` 和 `servo_bus.py` 不要混用。

Vosk 模型已随目录提供，标准口令转录无需 API 密钥。自由口语理解的密钥通过 `DEEPSEEK_API_KEY` 或 `ARM_LLM_API_KEY` 环境变量配置；`ARM_LLM_URL`、`ARM_LLM_MODEL` 可覆盖接口与模型。当前接口连通性与服务端模型可用性未在本次复查中验证。

## 关键未解决问题

- **底座反馈异常仍需定位。** 历史视觉回位中，底座从 720 请求到 950 时反馈跳到 0，随后整组停止并确认驱动关闭。见 [失败记录](so101_scs215_portable/verification/pointing_home_failure_20261002.json)。不能通过扩大限位、修改跨零方向或反复重试直接宣称解决。
- **1.2 的真实几何关系与精度尚缺完整证据。** [实机验证记录](so101_scs215_portable/verification/1_2_实机验证记录_20261001.md) 区分了成功的小范围动作、失败的底座路径和模型估算。需要先解决反馈与路径问题，再验证六方向、指定距离及外部测量误差。
- **视觉标定尚未完成。** 最新流程需现场完成四点实际指向并保存 `vision.pointing_calibration`；移动相机、屏幕或机械臂后重新标定。应检查中间位置、边缘、目标丢失和停止效果，不能以四点保存成功代替整屏跟随验收。
- [2026-10-02 最后停机记录](so101_scs215_portable/verification/pointing_photo_final_stop_20261002.json) 中六关节扭矩均为 0；这是历史快照，本次没有连接硬件复查。卡阻、振荡、异常噪声或失控时切断外接电源，停止成功须核对驱动关闭反馈。

## 本次复查与复现

本次仅进行离线复查，没有打开硬件串口、驱动实物、修改 EEPROM 或重新做现场标定。

| 检查 | 2026-10-12 结果 |
|---|---|
| 进阶工程 `test_offline.py` + `test_advanced.py` | 220 项全部通过，耗时 134.302 秒；包含真实 Vosk 模型处理合成录音 |
| 原位置作业 `test_offline.py` | 8 项全部通过 |
| 原位置作业六轴 `0.5` 干运行 | 通过；目标为 858、962、163、833、651、462，未打开串口 |
| Python 语法、JSON 解析 | 17 个 Python 文件、19 个 JSON 文件通过 |
| 本地 Markdown 链接、待提交文本密钥模式检查 | 通过，无断链或密钥模式命中 |
| 已复制文件与本地原文件比对 | 内容一致；进阶工程与最终课程 ZIP 中的 25 个对应文本文件也一致 |

在进阶工程目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest test_offline.py test_advanced.py
```

在原位置作业目录安装其依赖后运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest test_offline.py
.\.venv\Scripts\python.exe -X utf8 homework_control.py move 0.5 0.5 0.5 0.5 0.5 0.5 --dry-run
```

上述共 228 项离线测试通过，不能替代真实机械臂、现场麦克风和视觉闭环验收。历史 `verification/hackathon_checks.txt` 保留原日期与范围，不能当作 2026-10-12 的新验收结果。

## 第三方模型

`so101_scs215_portable/models/vosk-model-small-cn-0.22/` 是已有 Vosk 中文识别模型的原始文件，不是本项目训练成果。[Vosk 官方模型列表](https://alphacephei.com/vosk/models) 将该模型列为 Apache 2.0；随目录附带上游 [COPYING](so101_scs215_portable/models/vosk-model-small-cn-0.22/COPYING)。
