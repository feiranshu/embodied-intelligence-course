---
name: robot-arm-tools
description: 使用 SO-101 机械臂工具（姿态、动作、末端移动、关节角、视觉检测与追踪、动作模仿）完成用户的操作请求。用户要求机械臂运动、看东西、找物体、跟随物体或模仿人时使用。
---

# SO-101 机械臂工具使用规范

## 何时使用哪个工具
| 用户意图 | 工具 | 说明 |
|---|---|---|
| 回到某个已知姿态、休息、复位 | `go_to_pose` | 先查 `list_capabilities`；当前复位用“初始状态”，旧“立正”在 ID4 重录前不可用 |
| 做一套已录制的动作 | `run_action` | 只能用 `list_capabilities` 列出的动作 |
| 向某个方向移动若干厘米/毫米 | `move_end_effector` | 厘米换算成毫米；超过上限时分多次 |
| 小幅调整没有预设的姿态、张开/合上夹爪 | `set_joint_angles` | 只给需要改变的关节；大幅变化改用预设姿态 |
| 问“你看到了什么” | `look` | |
| 找到某个物体 / 对准它 / 跟着它 | `detect_object` → `start_tracking` | 找不到就告诉用户，不要盲动 |
| 学我的动作 | `imitate_person` | |
| 停、别动、危险 | `stop_robot` | 最高优先级 |

## 调用前检查
1. 不确定名称或范围 → 先 `list_capabilities`。
2. 末端移动或视觉追踪前 → `get_robot_state`，确认 `cartesian_ready` 为 true；否则先 `go_to_pose("初始状态")`。
3. 启动追踪前 → 确认已经有 `detect_object` 返回的 `object_id`，且 `found` 为 true。
4. 已有追踪在运行时 → 先 `stop_tracking` 再做其他运动。
5. 每次模型回复只调用一个运动工具，拿到结果后再规划下一步；不要用连续小步绕过关节变化限制。

## 必须停止的情况
- 用户说停止、别动、危险，或语气表明紧急。
- 工具连续两次返回 `ok: false` 且原因相同（例如超出工作范围、关节未到位）。
- `get_robot_state` 中追踪状态为 `error`、`target_lost` 或 `workspace_edge`。
- 追踪状态为 `travel_limit` 或 `diverging`，或工具提示手眼标定未独立验证。
- 用户的指令含义不清、涉及人或易碎物体附近的大幅运动：先停下来问清楚。

## 回复用户
说明做了哪些动作、结果如何；有关节被限位、目标丢失或动作失败时如实说明，不要声称完成了没有完成的动作。
