# SO-101 SCS215 实机联调记录 (`NOTES.md`)

## Stage 1 — 只读总线与摄像头发现（2026-11-22 20:15）
- **串口与协议**：`COM3`（Waveshare CH343，`1,000,000 baud`），Feetech SCS215 6 轴（ID 1–6，`8.9–9.1 V`，`14–15 °C`，全位置模式 `[0, 1023]`）。
- **机械臂摄像头**：`GENERAL WEBCAM`（DirectShow 索引 `1`，`640×480`，稳态帧间隔 `~0.047–0.063 s`）。已排除笔记本自带 `HD Camera`（索引 `0`）。

## Stage 2 — 最小实机动作 `gripper-check`（2026-11-22 20:22，已通过）
- 仅驱动 ID 6 夹爪：`390 → 388`（张开，最大负载 `180`）、`278 → 281`（闭合，最大负载 `180`），其余 5 轴保持断扭。
- 视频：`logs/gripper-check_20261122_202251.mp4`（76 帧），快照左上角可见白色夹爪张开/闭合。
- `torque_off_verified = true`。

## Stage 3 首次尝试排障与重大实机发现（2026-11-22 20:23）
- **现象**：按旧 `advanced.json` 预设 `初始状态 [950, 920, 300, 740, 80, 276]` 移动至 `ID 3 = 267` 时发生 ID 3 通信报错，看门狗自动切断全部扭矩，全部 6 轴安全断扭（`14–15 °C`，`9.0 V`）。
- **通过视频关键帧（`logs/hello_world_keyframes/frame_0061.png` → `frame_0185.png` → `frame_0245.png`）与 FK 对比定位根因**：
  - 旧 `advanced.json` 标称 `kinematics.directions = [1, -1, -1, 1, 1]` 的 **Z 轴方向与实物相反**；
  - 配套 `config/calibration_reference/hand_eye.json` 中的 **`calibrated_joint_mapping` FK/IK 与实物完全一致**：
    - `ID 3 = 60`（朝白板）：`z = 304.1 mm`（高处）
    - `ID 3 = 142`（`frame_0061.png`，清晰俯拍木质桌面）：`z = 205.8 mm`（距桌面约 15 cm）
    - `ID 3 = 222`（`frame_0185.png`）：`z = 71.0 mm`（距桌面约 1.5 cm）
    - `ID 3 = 267`（`frame_0245.png`，夹爪指尖轻触木桌）：**`z_table ≈ 55.5 mm`**
    - 旧预设 `初始状态`（`ID 3 = 300`）：`z = -27.5 mm`（位于当前桌面下方！）
- **已实施修复**：
  1. 在 `jarvis/hardware.py` 的 `HardwareMotion.__init__` 中自动加载 `calibrated_joint_mapping` 应用到 `arm.model`，确保所有 FK/IK、工作区和桌面防碰撞平面 `table_z_mm` 方向与实物一致。
  2. 将安全观察位更新为实测高空俯拍位 **`OBSERVE_POSE_RAW = [990, 905, 145, 700, 285, 280]`**（校准 `z = 198.0 mm`，距桌面约 14 cm），安全桌面底面硬限位设为 **`Z_TABLE_FLOOR_MM = 60.0 mm`**，安全停泊位设为 **`SAFE_REST_POSE_RAW = [975, 900, 240, 712, 185, 280]`**（校准 `z ≈ 75 mm`，高出桌面约 2 cm，不压桌面）。
  3. 抬升时强制先抬肘部 ID 3（`joint_order=(3, 2, 4, 5, 1, 6)`）脱离桌面后再调整姿态。

## 2026-11-23 更新 GitHub 标定与实机绿色笔盒抓取验收
- **GitHub 标定同步**：更新本地 `embodied-intelligence-course` 至 `31338cb`（12 组拟合 RMS `3.42 px`，8 组留出 RMS `36.39 px`），同步 `hand_eye.json`、`camera_intrinsics.json` 与 `servo_bus.py` 至 `jarvis_grasp`。
- **硬件与运动学排障修复**：
  1. 只读预检发现 ID 4 因 11-22 晚旋转采集未正常收尾而停留在 `wheel` 模式（`hw_limits=[0,0]`），已在断扭状态下恢复为 `position` 模式 `[0, 1023]`。
  2. 确认 11-22 晚已对 ID 1–4 舵盘重新居中（全非跨零区间：`ID1 141..868`、`ID2 78..711`、`ID3 279..860`、`ID4 104..758`、`ID6 260..662`），并通过 20 组标定实拍图实测验证 URDF 物理方向为 `directions=[1, 1, 1, 1, 1]`，更新了 `vendor/scs215/advanced.json` 与安全观察/停泊位。
  3. 修复 `ColorShapeDetector(real_world=True)` 在冷光暗色桌边（`V≈30–45`）的蓝色误检、木纹橙色误检及自动曝光下高亮度绿盒漏检（`V≤252, S≥44`）。
- **分阶段实机通过记录**：
  - **Stage 2 (`gripper-check`)**：ID 6 开 `414` / 闭 `274`，`torque_off_verified=true`（`logs/gripper-check_20261123_214241.mp4`）。
  - **Stage 3 (`hello-world`)**：安全抬头俯拍识别前方绿色笔盒，实测 ±15 mm Z/Y 笛卡尔到位误差 `0.6–2.3 mm`，回泊断扭通过（`logs/hello-world_20261123_214654.mp4`，541 帧）。
  - **Stage 4 绿色笔盒实机抓取成功 (`real_grasp_pouch.py --color green --grasp-z-mm 26.0`)**：
    - 视觉识别 → 张爪 `560` → 分段下降至 `z=24.9 mm` → 负载接触锁定闭合（`raw=335`, `max_load=480`, `stop_reason=load_contact`, `gripped=true`）→ 垂直抬升至 `z=160.5 mm` 并静置复检（`gripper_raw=331`, `gripper_load=195`, `visual_in_hand_area_px=223797.5`, `grasp_verified=true`，`logs/pouch_lifted.png` 确认笔盒悬空稳夹于双爪间）→ 轻放回桌面松爪（`raw=594`，`logs/pouch_after_release.png` 确认笔盒完好放回）→ 回安全停泊位并确认 6 轴全部断扭（`torque_off_verified=true`，`logs/real_grasp_20261123_215541.mp4`，1250 帧）。
