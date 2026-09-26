# SO-101 / SCS215 位置模式六轴作业

这个目录是从 `so101_scs215_portable` 中独立出来的作业版本，只保留以下功能：

1. 查看串口和六只 SCS215 舵机状态；
2. 修改舵机 ID；
3. 恢复位置模式并记录校准端点；
4. 输入六个 `0～1` 数值，让六只舵机同步移动到目标位置；
5. 单独测试一只舵机，避免第一次调试就驱动全部关节。

本目录不包含原工程的网页连续旋转控制器。作业演示入口是
`homework_control.py`。

## 1. 当前硬件与校准

- 舵机：6 × Feetech SCS215，已观察到模型编号 `8965`
- 转接板：Waveshare Bus Servo Adapter (A)，CH343
- 波特率：`1,000,000`
- 协议：SCS，16 位寄存器按高字节在前发送

当前 `config.json` 中保存的校准区间如下：

| ID | 关节 | 0.0 端点 | 1.0 端点 | 输入 0.5 的目标 |
|---:|---|---:|---:|---:|
| 1 | 底座 | 694 | 1022 | 858 |
| 2 | 肩部 | 902 | 1022 | 962 |
| 3 | 肘部 | 11 | 315 | 163 |
| 4 | 腕部俯仰 | 661 | 1005 | 833 |
| 5 | 腕部旋转 | 388 | 914 | 651 |
| 6 | 夹爪 | 266 | 659 | 462 |

这些区间采用“不拆装、缩小可用范围”的处理方式，均不跨越位置反馈的
`1023→0` 回绕点。重新装配、换舵机或改变舵盘角度后必须重新校准。

## 2. Windows 安装

推荐安装 Python 3.10 或更高版本，然后在本目录运行：

```powershell
.\setup_windows.bat
```

如果系统默认的 `python` 指向 MSYS，可以显式传入 Windows Python：

```powershell
.\setup_windows.bat "D:\anaconda3\python.exe"
```

以后打开新终端时激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

也可以不激活，直接使用：

```powershell
& ".\.venv\Scripts\python.exe" ".\homework_control.py" ports
```

## 3. 推荐操作顺序

列出串口（不访问舵机）：

```powershell
python homework_control.py ports
```

只读查看六只舵机的模式、位置和校准端点：

```powershell
python homework_control.py --port COM5 status
```

只有确实处于连续模式时才运行以下命令。它会把六只舵机 EEPROM 的角度
限位写成 `0/1023`，不应反复执行：

```powershell
python homework_control.py --port COM5 position-mode
```

校准某一只舵机只读取手动摆放位置，不主动驱动它：

```powershell
python homework_control.py --port COM5 calibrate 1
```

第一次测试一只舵机时，先干运行，再去掉 `--dry-run`：

```powershell
python homework_control.py --port COM5 move-one 1 0.75 --duration 3 --dry-run
python homework_control.py --port COM5 move-one 1 0.75 --duration 3
```

六轴中点目标的干运行：

```powershell
python homework_control.py --port COM5 move 0.50 0.50 0.50 0.50 0.50 0.50 --duration 4 --dry-run
```

确认模式、当前位置和机械路径安全后，去掉 `--dry-run` 才会实际运动：

```powershell
python homework_control.py --port COM5 move 0.50 0.50 0.50 0.50 0.50 0.50 --duration 4
```

六个数依次对应 ID 1～6。`--duration 4` 表示主机在约 4 秒内用平滑插值
逐步到达目标；改成 `3` 会更快，目标位置不变。

## 4. 离线验证

以下命令不访问串口：

```powershell
python -m unittest -v test_offline.py
python -m json.tool config.json
```

## 5. 安全限制

- 第一次运动前先用 `status`，并准备随时切断舵机外接电源。
- 不要同时运行网页控制器、串口调试软件和本程序；COM 口一次只能被一个进程占用。
- `move` 会同时驱动六只舵机；组合姿态是否自碰撞仍需操作者现场判断。
- 程序完成、中断或异常时会尝试释放扭矩，但软件停止不能代替物理断电。
- 出现卡阻、异响、反向运动或线缆受拉时，直接切断舵机外接电源。
- 当前线性位置方案不能表示跨越 `0/1023` 的物理行程；不要把跨回绕点的两个位置记录成同一校准区间。

## 6. 文件说明与提交内容

| 文件 | 作用 |
|---|---|
| `homework_control.py` | 作业主程序：编号、校准、状态、单轴和六轴位置控制 |
| `servo_bus.py` | SCS215 串口协议、寄存器访问和同步写位置 |
| `config.json` | 串口设置、关节名称和现场校准值 |
| `test_offline.py` | 不接硬件的协议与控制逻辑测试 |
| `requirements.txt` | Python 依赖，目前只有 `pyserial` |
| `作业说明.md` | 面向教师的实现原理说明 |
| `HANDOFF.md` | 面向后续维护者的技术交接 |

提交时建议把本目录的源码和文档连同演示视频一起压缩，但删除 `.venv`、
`__pycache__` 和其他本机临时文件。视频中至少展示 `status`、一条实际
`move` 命令、机械臂响应以及程序最后的反馈信息。

