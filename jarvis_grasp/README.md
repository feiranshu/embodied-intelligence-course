# JARVIS 桌面物体识别与抓取系统

基于当前 `project`（第 23 组 JARVIS 桌面寻物助手）与底层 `vendor/scs215`（SO-101 SCS215 六轴机械臂控制）扩展构建的独立软件包，完整实现**物体识别、模糊指令澄清、视觉闭环对准、三维抓取规划、分段下降抓取、夹持检测、抬起核对与放置释放**流程。

> **运行安全说明**：默认采用**软件模拟模式**（合成桌面与 2.5D 夹爪反馈），不会打开串口、摄像头或麦克风，也不会调用收费外部模型。实机模式和实机抓取分别受 `hardware.enabled`、`calibration.verified` 与 `hardware.grasp.verified` 三重安全闸门保护。

## 目录结构

| 路径 | 职责 |
| --- | --- |
| `jarvis/perception.py` | 物体识别（颜色/形状/朝向角、ArUco + PnP 3D 位姿）、多轮意图解析（识别/抓取/放下/停止）、模板跟踪与检测框叠加绘制 |
| `jarvis/grasp.py` | 抓取参数安全校验、单目像素-桌面射线求交 3D 定位（`ray_plane_intersection`）、抓取规划（`estimate_grasp_pose`）与分阶段抓取状态机（`GraspExecutor`） |
| `jarvis/controller.py` | 统一任务控制器：识别 → 模糊澄清 → 视觉居中与身份核对 → 抓取/放置状态机 → 急停与日志记录 |
| `jarvis/simulation.py` | 2.5D 合成桌面与平行夹爪模拟，支持空抓（`grasp_miss`）、抬起滑落（`grasp_slip`）等 7 种测试场景 |
| `jarvis/hardware.py` | SCS215 SO-101 实机封装：线程相机、工作区与桌面防碰撞平面（`table_z_mm`）检查、分段笛卡尔移动（`move_delta`）、带负载/堵转接触检测的夹爪控制（`set_gripper`）及可选手眼标定加载 |
| `jarvis/providers.py` | OpenAI 兼容视觉检测（`VisionDetector`）、多意图选择（`ModelSelector`）与语音转录接口 |
| `jarvis/server.py`、`web/index.html` | 本地网页演示界面：实时画面检测框、抓取流程进度条、夹爪状态、一键放下与本地停止 |
| `jarvis/grasp_demo.py` | 一键演示脚本（识别 → 澄清 → 抓取 → 放下） |
| `jarvis/calibrate_grasp.py` | SCS215 实机只读抓取标定助手（不开启扭矩，用于实测桌面高度与夹爪读数） |
| `jarvis/experiment.py`、`evaluate.py` | 离线批量评测与分组统计 |
| `config/default.json` | 默认软件模拟配置（实机入口关闭） |
| `config/hardware_example.json` | 实机配置模板（含手眼参考路径与抓取安全参数说明） |
| `config/calibration_reference/` | 第 6 周相机内参（`camera_intrinsics.json`）与经验手眼矩阵（`hand_eye.json`）参考 |
| `vendor/scs215/` | 原有 SO-101 SCS215 底层舵机通信、关节映射、URDF 运动学与逆解代码 |
| `tests/` | 离线单元测试（覆盖识别、澄清、3D 定位、抓取、空抓/滑落、急停、实机安全闸门与 Web API） |

## 快速开始（软件模拟）

需要 Python 3.10 及以上版本（已在 Python 3.12 验证）：

```powershell
cd jarvis_grasp
python -m pip install -r requirements.txt
```

### 1. 启动网页界面（推荐）

```powershell
python -X utf8 -m jarvis --web
```

打开 <http://127.0.0.1:8765>（或双击 `start_demo.bat`）。可直接尝试：
- `抓取蓝色盒子`：识别目标 → 视觉居中对准 → 张开夹爪 → 安全下降 → 闭合夹取 → 抬起核对。
- `抓取红色那个` → 追问“你要抓圆的还是方的？” → 回答 `圆的` → 自动完成抓取。
- `把绿色球抓起来放到左边`：抓起后水平移动并放置到左侧。
- 点击右上角 `放下物体` 或输入 `放下`：将当前夹持物体放回桌面并回到观察高度。
- 在右侧下拉框切换 `抓取故障 · 夹爪空抓测试` 或 `抬起过程物体滑落`，验证失败检测与安全退回。

### 2. 命令行与一键演示

```powershell
# 一键完整演示（识别 + 模糊澄清 + 抓取 + 放下）
python -X utf8 -m jarvis.grasp_demo --command "抓取红色那个" --answer "圆的"

# 单条抓取指令
python -X utf8 -m jarvis --once "抓取蓝色盒子"

# 多轮交互命令行
python -X utf8 -m jarvis
```

### 3. 接入视觉大模型与语音（可选）

```powershell
python -m pip install -r requirements-optional.txt
$env:JARVIS_BASE_URL = "OpenAI 兼容接口地址"
$env:JARVIS_API_KEY = "你的密钥"
$env:JARVIS_MODEL = "支持图像与 Function Calling 的模型名"
python -X utf8 -m jarvis --model --detector vision --web
```

模型只返回受限工具调用（目标框、候选编号与 `find`/`grasp`/`release`/`stop`/`chat` 意图），不拥有原始舵机控制权；所有轨迹插值、防碰撞检查与急停均由本地代码执行。

## 测试与离线评测

```powershell
# 运行全部单元测试
python -X utf8 -m unittest discover -s tests -v

# 运行 50 组离线识别与抓取评测（10 居中 + 10 抓取 + 10 对模糊抓取对照 + 5 不存在 + 5 故障）
python -X utf8 -m jarvis.experiment --output runs/my_experiment
python -X utf8 -m jarvis.evaluate runs/my_experiment/trials.jsonl --output runs/my_summary.json
```

## 实机部署与安全标定（SCS215 SO-101）

当前 `config/default.json` 默认关闭实机：
- `hardware.enabled = false`
- `hardware.calibration.verified = false`
- `hardware.grasp.verified = false`

接入真实 SO-101 机械臂与腕部相机前，请按以下步骤验证：

1. **只读采样高度与夹爪行程**（断电或关闭扭矩后手动摆放，不写入舵机）：
   ```powershell
   python -X utf8 -m jarvis.calibrate_grasp --port COM3
   ```
2. **复制并填写实机配置**：复制 `config/hardware_example.json` 为独立配置文件，填入实测的：
   - `image_to_base_mm`（3×2 归一化图像误差到基座毫米位移矩阵，需实测核对正负方向）
   - `workspace_mm`（基座 X/Y/Z 已验证安全上下界）
   - `grasp.table_z_mm`（桌面防碰撞下界，代码严格禁止下降低于该高度）
   - `grasp.grasp_z_mm`、`pre_grasp_z_mm`、`lift_z_mm`、`camera_to_gripper_mm`
   - `gripper_open_raw`、`gripper_close_raw`、`gripper_held_range_raw`（SCS215 ID 6 夹爪范围 `[273, 662]`，读数越大夹爪越张开）
3. 确认全部实测无误后，将 `enabled` 与两个 `verified` 置为 `true`，并在可随时断电的前提下启动：
   ```powershell
   python -X utf8 -m jarvis --hardware --config config/my_hardware.json --web
   ```

实机夹爪闭合采用**负载 + 位置停滞监测**：当夹爪接触物体导致位置不再前进或舵机负载达到 `gripper_max_load` 时，立即锁定轻微夹紧目标位，避免过流堵转；并根据稳定读数是否落在 `gripper_held_range_raw` 内判定是否空抓或滑落。任何异常或急停都会锁存停止状态，须排查后重启程序。
