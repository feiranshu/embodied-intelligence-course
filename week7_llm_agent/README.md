# 第七周：LLM Agent 课堂练习与课后作业

人工从零操作请先阅读 [`人工操作指南.md`](人工操作指南.md)；实现原理、测试证据、完成结果和已知限制见 [`第七周_LLM_Agent_报告.md`](第七周_LLM_Agent_报告.md)。

本目录对应[课程网页《EmbodiedAI Week 7: LLM Agent》“四、课堂练习 & 课后作业”](https://sa87gcdhzg.feishu.cn/wiki/LVVow0AxIiPbGDk5zedcuGaLnqc)的三项任务。机械臂控制复用随附的 `week4/so101_scs215_portable`，视觉伺服复用 `week6_visual_servo`；随附 week4 配置会随本周实机状态做安全更新，不会改动项目中的 GitHub 同步目录。

| 部分 | 子文件夹 | 任务 | 入口 |
| --- | --- | --- | --- |
| 4.1 | `part1_imitation` | 任务书任务一～四：拍照 → VLM 解析动作 → 6 关节角 → 安全姿态模仿；任务四改进 prompt | `imitate.py` |
| 4.2 | `part2_function_calling` | 任务五及扩展：用 Function Calling 调用已录制的连贯动作和此前功能；选做 Skill | `agent.py`、`skills/robot-arm-tools/SKILL.md` |
| 4.3 | `part3_omni_agent` | 腕部相机＋麦克风 → ASR＋视觉模型 → 语音对话并自主控制机械臂 | `omni_agent.py` |

公共代码：

| 文件 | 作用 |
| --- | --- |
| `llm_client.py` | 智谱 / 硅基流动 / 阿里云百炼 / DeepSeek 的 OpenAI 兼容客户端；API Key 只从环境变量读取；默认关闭 thinking |
| `arm_backend.py` | 统一的机械臂接口。`--execute` 时走 week4 的 `Arm` + `CommandRunner`（限位、跨零、反馈、出错关扭矩）；否则为模拟器，按同一份 `advanced.json` 的映射限位和 IK 校验 |
| `robot_tools.py` | 12 个工具的 JSON Schema 和实现（见下表） |
| `agent_loop.py` | LLM ⇄ 工具循环，支持非流式、流式（全模态模型必须流式）和不支持 Function Calling 时的文本协议 |
| `camera_hub.py` | 后台持续读帧的摄像头；也可用一张照片代替摄像头 |

## 0. 环境与 API Key

```powershell
# 从课程项目根目录启动已经配置好的虚拟环境
& .\.venv\Scripts\Activate.ps1
cd .\第7周_LLM_Agent_代码_20261019\第7周_LLM_Agent\week7_llm_agent
python -m pip check

# 仅在新电脑或重建环境时安装
python -m pip install -r requirements.txt
```

本机的依赖安装在课程根目录 `.venv` 中。运行本作业前必须先激活它，或直接使用该目录的 `python.exe`；未激活时系统的 `python` 当前会指向 MSYS2，缺少本作业依赖。

4.1～4.3 默认共用硅基流动 Key。API Key 放在环境变量里，不要写进代码或提交到仓库；遮蔽输入和持久化命令见 `人工操作指南.md`。只检查是否已经加载：

```powershell
if ($env:SILICONFLOW_API_KEY) { "SILICONFLOW_API_KEY 已配置" } else { "尚未配置" }
```

可用 `--provider zhipu|siliconflow|dashscope|deepseek` 切换平台，`--model` 指定模型，`--base-url` 覆盖地址。

**所有入口缺省都是模拟模式，不打开串口。** 加 `--execute` 才驱动实机，启动前要输入 `RUN` 确认。实机前请先按 week4 交接说明运行只读诊断 `diagnose.py`，并保证一次只有一个程序占用串口和摄像头；运行中按 Ctrl+C 会调用停止。week7 的六关节目标若从实时起点判断为全程无跨零，会沿用已实测顺滑的 `move_servo_raw.py` 控制方式：不改PID、只给实际运动的关节上扭矩，并用一个广播包写入各轴最终位置、3000 ms和speed 100；只要有一个关节需要跨零，就回退到原有的逐关节安全路径。

**ID4 当前状态（2026-10-22）：** 腕部俯仰舵机调整了物理方向，用户现场确认位置端点为 `104 → 758`，且不再跨零。本副本的 `advanced.json` 已据此更新 ID4 映射和原始限位。旧的 IK、笛卡尔移动及视觉追踪标定依赖调整前的安装方向，因此配置暂设 `kinematics.calibrated=false`，相关工具不会暴露；重新标定前只使用当前范围内的预设姿态。旧“立正”姿态的 ID4 原始值为 `986`，已超出新范围，程序会自动隐藏并拒绝执行，需现场重新录制后才能恢复。现场手摆的举手姿态 `[1015, 26, 940, 484, 946, 386]` 保存为 `举手_原始记录`；因ID2/ID3跨零切换受重力影响，自动执行使用同侧近似姿态 `举手=[1015, 1011, 15, 484, 946, 386]`。

无跨零的week7同步姿态使用 `sync_duration_s=3.0` 和 `sync_speed=100`，与操作者确认顺滑的原始位置命令一致。到位判定使用 `sync_release_tolerance=10`：六轴首次同时进入目标 ±10 格就结束并释放扭矩，不再要求连续三次稳定，以减少负载关节在目标附近反复修正。此前试验的50 Hz主机端插值和speed 500已撤回，因为实机观感更抖。程序结束时仍会关闭并复核全部扭矩。

本次队友打包的 `week4/so101_scs215_portable` 没有包含 `diagnose.py`。只读诊断请从项目的 GitHub 同步目录运行：`embodied-intelligence-course-main/week4/so101_scs215_portable/diagnose.py`；不要把诊断误当作重新标定。

## 4.1 让机械臂模仿你的动作

```powershell
# 任务一：拍一张照片（本机已确认机械臂摄像头当前为索引 1）
python part1_imitation/imitate.py capture --camera 1
# 任务二：VLM 描述照片中的手臂动作（打印模型输出）
python part1_imitation/imitate.py describe --camera 1 --provider siliconflow --model "Qwen/Qwen3.5-4B"
# 任务三：一条命令完成拍照 → Qwen → 关节角 → 已记录安全姿态；先模拟
python part1_imitation/imitate.py imitate --camera 1 --provider siliconflow --model "Qwen/Qwen3.5-4B" --prompt detailed
# 对应实机版本；按提示输入 RUN
python part1_imitation/imitate.py imitate --camera 1 --provider siliconflow --model "Qwen/Qwen3.5-4B" --prompt detailed --execute --port COM5
# 使用已经拍好的照片：模拟与实机版本
python part1_imitation/imitate.py imitate --image ".\runs\capture_实际时间\photo.jpg" --provider siliconflow --model "Qwen/Qwen3.5-4B" --prompt detailed
python part1_imitation/imitate.py imitate --image ".\runs\capture_实际时间\photo.jpg" --provider siliconflow --model "Qwen/Qwen3.5-4B" --prompt detailed --execute --port COM5
# 任务四：改进后的 prompt（默认），--repeat 可连续多次按回车拍照模仿
python part1_imitation/imitate.py imitate --camera 1 --provider siliconflow --model "Qwen/Qwen3.5-4B" --prompt detailed --execute --port COM5 --repeat
```

每次运行在 `runs/<任务>_<时间>/` 保存 `photo.jpg` 和 `log.json`（完整 prompt、模型原始回复、解析结果、安全处理说明、目标和到达角度），方便向助教展示。

实机输入 `RUN` 后，程序默认显示 `3、2、1` 倒计时，并在倒计时结束时冻结一帧；看到“已拍照，可以放下手臂”后即可结束真人姿势。可用 `--capture-delay 5` 延长准备时间，或用 `--capture-delay 0` 关闭倒计时。已有照片 `--image` 不等待。

**关节表示。** 模型输出的是每个关节在 `advanced.json` `joint_mapping` 坐标下的角度（度），程序换算回舵机原始读数；超出范围的值会被限制在范围内并打印说明。这样模型不需要处理 SCS215 的 0/1023 跨零问题。

**模仿安全策略。** VLM 给出的六关节角只用于判断它最接近哪个 prompt 示例，原始值、限位结果和距离都会写入日志；真正执行的是 `imitation_examples.json` 引用的已记录姿态，而不是任意拼出的六关节组合。这保留了“照片 → VLM → 关节参数 → 机械臂”的完整流程，同时避免“每个关节都没超限、组合起来却会碰撞”的风险。任务四验收时应先在模拟模式确认识别结果，再选择两种现场已核对的示例姿态演示。

模型还必须返回 `arm_visible=true`。画面中没有完整可辨认的手臂、模型返回 `matched_pose=无`，或遗漏可见性字段时，程序保存日志并拒绝运动，不再用默认关节角强行选择“最近姿态”。

执行前还会按实时位置检查六轴路径。`姿态2` 已按 ID2 的实际有限活动范围重新录制为 `[975, 995, 15, 476, 926, 378]`，不再使用临时的“姿态2 → 举手”替代。若将来某个目标需要跨零，只有在 `imitation_examples.json` 人工指定并核对了 `no_cross_fallback` 时才会替代；否则程序直接拒绝运动，不会任意选择另一姿态。

**任务四的 prompt 改进**（`part1_imitation/prompts.py` 的 `detailed_prompt`，素材在 `imitation_examples.json`）：

1. 逐个关节说明结构、作用、范围和已记录姿态的参考值；
2. 给出“人体部位 → 机械臂关节”的对应规则（单臂、镜像、看不清时保持参考值）；
3. 给出“动作 → 姿态名 → 关节角”示例，数值在运行时由 `advanced.json` 中仍符合当前限位的已记录姿态（举手、姿态2、姿态3、居中、左侧、右侧）换算，配置更新后示例自动同步；模型明确返回有效 `matched_pose` 时优先采用语义匹配，避免仅凭角度距离把“举手”误选成“姿态3”；
4. 要求按步骤推理，并输出最接近的示例 `matched_example`，便于检查模型的判断依据。

`imitation_examples.json` 里的关节方向说明是根据已记录姿态和正运动学推断的，**须到现场逐个关节核对**；如果实际方向不同，改这个文件即可，不用改代码。为了允许实机执行，新动作要先按 week4 流程记录成 `advanced.json` 中的姿态，再以 `{"action": "...", "pose": "姿态名"}` 加进 `examples`。直接填写 `joints` 的条目只能作为 prompt 参考，不会被本安全策略选作执行目标。

## 4.2 把之前的功能作为 Agent 的 functions

```powershell
# 纯机械臂工具（不接相机），交互式对话
python part2_function_calling/agent.py
python part2_function_calling/agent.py --execute --port COM5
# 只读验收：代码层隐藏全部运动工具
python part2_function_calling/agent.py --read-only --once "请调用工具读取机械臂状态"
# 单句演示，加载选做 Skill：模拟与实机版本
python part2_function_calling/agent.py --skill --once "请挥手"
python part2_function_calling/agent.py --skill --once "请挥手" --execute --port COM5
# 任务五：文本交给 Qwen，由模型选择连贯动作工具；缺省为模拟
python part2_function_calling/agent.py --provider siliconflow --model "Qwen/Qwen3.5-4B" --once "请挥手"
python part2_function_calling/agent.py --provider siliconflow --model "Qwen/Qwen3.5-4B" --once "请点头"
# 实机执行；按提示输入 RUN
python part2_function_calling/agent.py --provider siliconflow --model "Qwen/Qwen3.5-4B" --once "请挥手" --execute --port COM5
python part2_function_calling/agent.py --provider siliconflow --model "Qwen/Qwen3.5-4B" --once "请点头" --execute --port COM5
# 接上腕部相机，开放视觉工具；--person-camera 是给 imitate_person 拍人的相机
python part2_function_calling/agent.py --camera "GENERAL WEBCAM" --person-camera "GENERAL WEBCAM"
# 实机
python part2_function_calling/agent.py --camera "GENERAL WEBCAM" --execute
```

终端会逐条打印 `→ 调用工具 name(args)` 和 `← 结果 {...}`，即“LLM 提出调用 → 程序执行 → 结果返回 LLM → LLM 回答”的完整流程；`runs/agent_*/tool_calls.json` 和 `transcript.json` 保存全部调用和对话。

| 工具 | 来源 | 作用 |
| --- | --- | --- |
| `get_robot_state` | week4 FK | 关节角、末端坐标、能否做末端运动、追踪状态 |
| `list_capabilities` | `advanced.json` | 可用姿态、动作、关节范围、单次移动上限 |
| `go_to_pose` | week4 1.1 | 运动到已记录姿态（“初始状态”走 week4 的复位路径） |
| `run_action` | week4 1.1 | 执行已录制姿态或单关节关键帧组成的动作序列 |
| `move_end_effector` | week4 1.2 IK | 末端沿前后左右上下直线移动，单次 ≤ 50 mm |
| `set_joint_angles` | 本周 4.1 | 小幅调整部分或全部关节角；超范围自动限制，单次变化过大则拒绝 |
| `look` | VLM | 拍腕部相机画面并回答问题 |
| `detect_object` | VLM / ArUco | 按文字描述定位物体，返回 `object_id` |
| `start_tracking` / `stop_tracking` | week6 第三、四部分 | 以检测框初始化光流跟踪，通过手眼矩阵控制末端；累计步进默认最多 60 mm，误差连续增大时停止 |
| `imitate_person` | 本周 4.1 | 完整的拍照模仿流程 |
| `stop_robot` | week4 | 停止所有运动和追踪 |

核心代码：`robot_tools.py`（Schema + 实现）、`agent_loop.py`（`run_turn`）。系统 prompt 在 `part2_function_calling/system_prompt.md`。

代码层安全约束不依赖模型自觉：同一条模型回复若包含两个或更多运动工具，这些运动会全部拒绝；任一运动工具失败后，本轮立即结束，不允许模型自行追加恢复运动；`set_joint_angles` 仅允许小幅变化；追踪设累计行程上限并检测发散。当前 `hand_eye.json` 明确标记 `independent_accuracy_validated: false` 且弱关节方向未识别，因此**实机视觉追踪保持锁定**，模拟追踪仍可用于联调。任务五的两个动作都会先到达可执行基准 `[1008, 902, 130, 472, 893, 274]`；随后“挥手”让 ID1 在 `929 ↔ 1009` 间往返，“点头”让 ID4 按 `522 → 422 → 522 → 472` 运动，单关节往返阶段的其他五轴保持基准值。重新采集时 ID2 的反馈为 `900`，因低于确认端点 `902` 两格，命令目标采用合法端点 `902`。

**选做 Skill**：`part2_function_calling/skills/robot-arm-tools/SKILL.md`，说明何时用哪个工具、调用前检查什么、哪些情况必须停止；`--skill` 时追加到系统 prompt。

如果所选模型不支持 Function Calling，加 `--protocol text`：工具说明写进 prompt，模型输出 `<tool_call>{...}</tool_call>`，由程序解析执行。

## 4.3 视觉-语音端到端 Agent

```powershell
# 默认：硅基流动 SenseVoiceSmall ASR + Qwen3.5-4B，模拟机械臂
python part3_omni_agent/omni_agent.py --protocol text --camera 1
# 仅检查相机和麦克风；不需要 API Key、不连接机械臂
python part3_omni_agent/omni_agent.py --check-devices --camera 1
# 实机（确认模拟正确后）
python part3_omni_agent/omni_agent.py --protocol text --camera 1 --execute --port COM5
# 没有麦克风时用键盘代替语音调试（仍然发送相机画面）
python part3_omni_agent/omni_agent.py --text --protocol text --camera 1
python part3_omni_agent/omni_agent.py --text --protocol text --camera 1 --execute --port COM5
```

每一轮：按回车开始说话，说完再按回车 → 硅基流动 ASR 把 WAV 转成文字 → 程序把识别文字和腕部相机画面发给视觉模型，并附上 4.2 的全部工具 → 模型选择工具 → 回复文字在终端流式打印并用本地 TTS 朗读。为避免小模型完成动作后擅自复位，4.3 每轮最多成功执行一个运动工具；连贯动作封装在单个 `run_action` 中。每轮录音和对话记录保存在 `runs/omni_*/`。

默认只需已经用于 4.1/4.2 的 `SILICONFLOW_API_KEY`。推荐为小模型加 `--protocol text`，让工具调用不依赖模型的原生 Function Calling。若使用百炼全模态模型，可加 `--provider dashscope --model qwen3.8-omni-flash --audio-input native`，直接发送原始音频。

**模型选择说明（2026-10 百炼文档）**：`qwen3.8-omni-flash` 支持图像＋音频输入和 Function Calling，但只输出文本，所以用本地 TTS 朗读（优先 `pyttsx3`，没有则用 Windows 自带的 System.Speech）。`qwen3.5-omni` 和 `qwen3-omni-flash` 能直接输出语音，但不支持 Function Calling；使用它们时加 `--model qwen3.5-omni-flash --protocol text --audio-reply`，由文本协议调用工具、播放模型自己的语音。

## 离线自检

```powershell
python -X utf8 -m unittest test_offline.py -v
```

离线测试使用伪造的 LLM 回复、合成图片和模拟机械臂，覆盖：JSON 解析、关节角与原始读数互换及限位、4.1 安全姿态流程、全部工具规则、同轮多运动拒绝、相机工具可见性、非流式/流式/文本协议三种 Agent 循环、轮数上限、VLM 检测框 → 光流跟踪 → 视觉伺服限程，以及 4.3 的历史压缩和音频编码。具体通过数量以测试命令最后一行输出为准。

这些测试**不访问真实 API、摄像头、麦克风或串口**，通过不代表模型效果或实机运动已验收。验收前需在现场：配置 API Key 并实际调用一次；核对 `imitation_examples.json` 中的关节方向；用模拟模式确认模型输出合理后再加 `--execute`。
