# SO-101 SCS215 可移植实时控制器

课程基础与进阶任务（预设姿态/动作、URDF IK、键盘直线移动、中文语音、黑球视觉闭环）的安装、模拟、原理和实机标定方法见 [进阶使用说明.md](进阶使用说明.md)。

这是当前六关节机械臂的独立交接包。它不依赖 LeRobot，也不依赖原电脑上的 Conda 路径；运行时只需要 Python、`pyserial`、Waveshare 总线舵机转接板及舵机外接电源。

## 当前硬件

- 电机：6 × Feetech SCS215，总线返回型号编号 `8965`
- USB 转接板：Waveshare Bus Servo Adapter (A)，CH343
- 通信：SCS 协议，`1,000,000 baud`
- 默认 ID：

| 按键/ID | 关节 | 配置名称 |
|---:|---|---|
| 1 | 底座 | `shoulder_pan` |
| 2 | 肩部 | `shoulder_lift` |
| 3 | 肘部 | `elbow_flex` |
| 4 | 腕部俯仰 | `wrist_flex` |
| 5 | 腕部旋转 | `wrist_roll` |
| 6 | 夹爪 | `gripper` |

## 新电脑首次使用（Windows）

1. 将整个文件夹复制到新电脑，不要只复制单个 `.py` 文件。
2. 安装 Python 3.10 或更高版本。安装时勾选“Add Python to PATH”。
3. 安装 CH343 驱动，使设备管理器中出现 `USB-Enhanced-SERIAL CH343 (COMx)`。
4. 接好六只舵机总线、USB 和舵机外接电源。
5. 双击 `setup_windows.bat`。它会在本目录创建隔离的 `.venv` 并安装依赖。
6. 双击 `diagnose_windows.bat`。确认 ID 1～6 全部出现后，再启动控制器。
7. 双击 `start_local.bat`，浏览器将自动打开控制页面。

串口默认是 `auto`，程序会通过 CH343 的 VID/PID 自动选择端口，因此换电脑后不需要继续使用 `COM8`。如果电脑同时连接多个 CH343，请在 `config.json` 中把：

```json
"port": "auto"
```

改为实际端口，例如：

```json
"port": "COM5"
```

## 实时控制

- 数字键 `1～6`：选择关节。
- 按住 `←` 或 `A`：向左旋转。
- 按住 `→` 或 `D`：向右旋转。
- 松开方向键：停止当前舵机并释放扭矩。
- 空格：停止全部舵机并释放扭矩。
- 网页失去焦点或切到后台：请求停止全部舵机。
- 浏览器运行时每 250 ms 发送心跳；约 0.8 秒收不到心跳时，主机自动停止当前舵机。

首次使用应把速度保持在 `100`，用短按检查每个关节。界面的“连续模式/位置模式”来自舵机 EEPROM 实际读数，不是软件猜测。

“全部切换连续模式”会把六只舵机的最小/最大角度限制持久写成 `0/0`；“全部恢复位置模式”会写成 `0/1023`。模式写入会在断电后保留。

如果某个关节的界面左右与实际期望相反，在 `config.json` 中把该舵机的：

```json
"invert_direction": false
```

改为：

```json
"invert_direction": true
```

## 通过另一台电脑或手机控制

控制板必须通过 USB 接在运行服务的主机上，舵机也必须由该主机旁边的转接板和外部电源连接。另一台设备只是通过局域网打开网页。

1. 让两台设备连接同一个可信局域网。
2. 在连接机械臂的主机上双击 `start_lan.bat`。
3. Windows 防火墙首次询问时，只允许“专用网络”。
4. 终端会打印形如 `http://192.168.x.x:8767/?token=...` 的地址。
5. 在另一台电脑或手机打开完整地址（必须包含 `token`）。

每次 LAN 启动都会生成新令牌。不要把这个地址转发给无关人员。默认 `start_local.bat` 仅监听 `127.0.0.1`，局域网无法访问。

## Conda 用户

也可以不用 `.venv`：

```bat
conda activate lerobot-so101
python -m pip install -r requirements.txt
python diagnose.py
python controller.py
```

LAN 模式：

```bat
python controller.py --lan
```

临时指定端口：

```bat
python controller.py --port COM8
```

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python diagnose.py
python controller.py
```

局域网控制使用 `python controller.py --lan`。在 Linux 上，当前用户还需要拥有串口权限；设备名通常类似 `/dev/ttyUSB0`。在 macOS 上通常类似 `/dev/cu.usbserial-*`。自动识别失败时用 `config.json` 或 `--port` 指定。

## 停止和退出

在网页按空格，然后回到控制器终端按 `Ctrl+C`。正常退出时程序会：

1. 向六个 ID 发送速度归零；
2. 请求关闭六个舵机扭矩；
3. 关闭串口。

网页停止命令和软件退出不能代替物理断电。出现卡阻、异常噪声、失控或线缆缠绕时，直接切断舵机外接电源。

## 目录说明

```text
config.json             硬件、串口、关节名称和方向配置
controller.py           本机/LAN 网页服务入口
servo_bus.py            SCS215 协议与舵机控制实现
diagnose.py             严格只读的串口、模式和位置检查
web/index.html          网页界面
setup_windows.bat       新电脑安装依赖
diagnose_windows.bat    Windows 只读检查入口
start_local.bat         本机控制入口
start_lan.bat           局域网控制入口
HANDOFF.md              给维护者或其他 agent 的技术交接
test_offline.py         不接硬件即可运行的协议/配置回归测试
```

## 常见问题

### `PermissionError(13, 拒绝访问)`

COM 端口正被其他程序占用。关闭旧控制器、VS Code 串口监视器、Feetech 调试软件以及仍在运行的 Python 进程，再重新启动。一个串口同一时刻只能被一个程序打开。

### 找不到 CH343

检查 USB 数据线、设备管理器和 CH343 驱动。运行 `diagnose_windows.bat` 查看所有串口。如果存在多个 CH343，在 `config.json` 中写死正确的 `COMx`。

### ID 无回复

先切断舵机电源，检查总线插头方向、三线连接、串联线路和外接电源，再重新上电。不要在插头错位或短路时反复发送命令。

### 网页能打开但另一台设备打不开

确认运行的是 `start_lan.bat`，两台设备在同一局域网，并且 Windows 防火墙允许 Python 访问专用网络。必须使用终端打印的完整 IP、端口和令牌地址。
