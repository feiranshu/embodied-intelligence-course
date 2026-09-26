# 位置模式作业技术交接

更新日期：2026-09-26

## 1. 目录来源与目标

本目录源自 `so101_scs215_portable_2026-09-22.zip`。原包的主要用途是网页
控制一只 SCS215 舵机以连续模式转动；本目录只交接后来增加的课程作业功能：

- 舵机 ID 1～6 的编号；
- 位置模式恢复；
- 两端点校准；
- 六个 `0～1` 输入到六个原始位置的映射；
- 广播同步写入六个目标，使六轴一起运动；
- 单轴位置运动测试与离线回归测试。

原包中的 `controller.py`、`web/`、LAN 服务及连续旋转页面没有复制到这里。

## 2. 相对原始压缩包的改动

### 新增

- `homework_control.py`：作业命令行入口，包含 `ports`、`status`、
  `set-id`、`position-mode`、`calibrate`、`move-one` 和 `move`。
- `作业说明.md`：课程提交用的原理、操作和安全说明。
- 本独立目录中的 `README.md`、`HANDOFF.md`、`AGENTS.md`。

### 修改

- `servo_bus.py`
  - 增加 `SERVO_ID=0x05`、`GOAL_POSITION=0x2A`、广播 ID `0xFE` 和
    `SYNC_WRITE=0x83`；
  - 增加无需状态回复的广播写入；
  - 增加 `sync_write_positions()`，用一个包同时携带多个舵机目标；
  - 增加 `set_servo_id()`，完成解锁、改 ID、重新上锁和新 ID 通信验证。
- `test_offline.py`
  - 从原来的 3 项测试扩展到 8 项；
  - 增加同步写包字节序、归一化范围、反向区间、插值终点和单轴隔离测试；
  - 测试使用自己的固定校准夹具，不依赖现场后来修改的端点。
- `config.json`
  - ID 1～4 已由操作者通过 `calibrate` 重新记录；
  - ID 5～6 保留原包数值；
  - JSON 因原子写回被格式化为多行。
- 原工程 `README.md`
  - 增加作业入口、干运行与 `move-one` 说明；
  - 原有网页连续模式说明保留。

### 未改动的原包内容

`controller.py`、`diagnose.py`、`web/index.html`、原 `HANDOFF.md`、
`requirements.txt`、启动批处理和 `.gitignore` 的控制逻辑均未因作业功能而改写。
本独立目录只复制其中仍被位置作业需要的依赖文件。

## 3. 当前校准与回绕问题

实物位置反馈在 `1023` 与 `0` 之间回绕，而且机械结构不能沿另一侧旋转一整圈。
原先曾观察到类似：

```text
696 → ... → 1022 → 0 → ... → 306
```

当前位置模式代码使用普通线性公式：

```text
raw = round(end_a + x * (end_b - end_a))
```

因此不能把跨回绕点的两个物理端点直接作为 `end_a/end_b`。当前采用的临时方案
是缩小每个关节的可用范围，使两个端点位于回绕点同一侧：

| ID | 关节 | end_a | end_b | config 中的 home |
|---:|---|---:|---:|---:|
| 1 | 底座 | 694 | 1022 | 858 |
| 2 | 肩部 | 902 | 1022 | 962 |
| 3 | 肘部 | 11 | 315 | 163 |
| 4 | 腕部俯仰 | 661 | 1005 | 833 |
| 5 | 腕部旋转 | 388 | 914 | 651 |
| 6 | 夹爪 | 266 | 659 | 463 |

`home` 当前只作为记录保存，`move` 实际按端点公式计算；例如 ID 6 的
`0.5` 在 Python 舍入规则下为 `462`。长期若要恢复跨回绕点的完整机械行程，
需要改成连续模式下由主机读取反馈并闭环控制，而不是继续扩展此线性位置方案。

## 4. 协议与运动实现

SCS 数据包：

```text
FF FF ID LENGTH INSTRUCTION PARAMS... CHECKSUM
CHECKSUM = ~(ID + LENGTH + INSTRUCTION + PARAMS...) & 0xFF
```

位置作业使用的寄存器：

| 地址 | 用途 |
|---:|---|
| `0x05` | 舵机 ID（EEPROM） |
| `0x09` / `0x0B` | 最小/最大位置限位 |
| `0x28` | 扭矩开关 |
| `0x2A` | 目标位置 |
| `0x30` | EEPROM 锁 |
| `0x38` | 当前位置 |

实物 SCS215 使用 1 Mbps，16 位寄存器按高字节在前。六轴运动流程为：

1. 验证六个输入都在 `0～1`；
2. 验证六只舵机都为位置模式，且起点位于各自校准区间附近；
3. 先将当前位置写为目标，避免上扭矩后追赶旧目标；
4. 分别打开六只舵机扭矩；
5. 主机按 smoothstep 在指定 `duration` 内插值，默认 25 Hz；
6. 每一步使用一个广播同步写包携带 ID 1～6 的目标位置；
7. 完成、异常或 `Ctrl+C` 时尽力释放六只舵机扭矩。

`move-one` 只在广播包中放入选定 ID，并在上扭矩前读取 5 次位置；读数跨度
超过 15 ticks 会拒绝运动。六轴 `move` 当前对每只舵机只读取一次起点，这是
一个已知的安全性差异。

## 5. 已知限制

- `_read_status()` 校验帧头、长度、校验和与错误码，但尚未核对回复 ID 是否与
  请求 ID 一致。此前 ID 1 曾出现一次偶发异常读数，后续若继续增强可靠性，
  应优先补回复 ID 校验和多次采样。
- `status` 对每只舵机只显示一次位置读数，不代表统计稳定性。
- `position-mode` 当前一次把 ID 1～6 的 EEPROM 限位都写成 `0/1023`，没有
  单 ID 子命令；不要为普通运动重复执行。
- 校准的机械安全性依赖操作者选择端点；软件无法判断组合姿态是否自碰撞。
- `sync_write_positions()` 是广播写，不会收到每只舵机的确认包，最终位置反馈
  只能在轨迹结束后逐只读取。
- 本目录中的 `servo_bus.py` 保留少量原工程连续模式方法，以复用已验证的协议
  实现；`homework_control.py` 的位置作业不会调用网页控制器。

## 6. 接手验证

默认只做离线验证，不连接实物：

```powershell
python -m py_compile homework_control.py servo_bus.py
python -m unittest -v test_offline.py
python -m json.tool config.json
python homework_control.py move 0.5 0.5 0.5 0.5 0.5 0.5 --dry-run
```

只有操作者明确表示人在机械臂旁、路径无障碍且能立即断开外接电源时，才执行
去掉 `--dry-run` 的命令。实物 EEPROM 模式和实时位置不能仅凭本文档假定，
应先运行只读的：

```powershell
python homework_control.py --port COM5 status
```

## 7. 最近验证状态

2026-09-26 建立独立目录后已经完成以下离线验证：

- `homework_control.py`、`servo_bus.py` 语法编译通过；
- `python -m unittest -v test_offline.py`：8 项全部通过；
- `config.json` JSON 解析通过；
- 六轴 `0.5`、`--duration 4` 干运行通过，目标为
  `858, 962, 163, 833, 651, 462`。

以上验证均未打开串口。实际硬件模式与当前位置仍以现场 `status` 输出为准。
