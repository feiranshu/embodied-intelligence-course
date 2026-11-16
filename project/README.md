# JARVIS：会追问的桌面寻物助手

依据第 23 组《JARVIS项目proposal》实现的独立软件项目。原提案正文保存在 `docs/proposal.txt`。所有代码、依赖配置、演示页面和运行记录都在本目录，运行不依赖旁边的课堂练习文件夹。

**默认是软件模拟，不会打开串口、摄像头或麦克风，也不调用外部模型。** 合成画面中的物体经过真实图像检测和跟踪；相机运动是简化的二维反馈模型，不是 MuJoCo 动力学仿真，不代表机械臂精度、碰撞安全或实机成功率。

## 运行

需要 Python 3.10 或更新版本。本次使用 Python 3.12 验证。

```powershell
cd project
python -m pip install -r requirements.txt
python -X utf8 -m jarvis --web
```

打开 <http://127.0.0.1:8765>。也可双击 `start_demo.bat`。网页只监听本机。

演示顺序：

1. 输入“帮我看看蓝色盒子”，观察画面居中、保持一秒及最终核对。
2. 点击“重新开始”，输入“红色那个”，系统追问后回答“圆的”。等待回答期间不会运动。
3. 输入“紫色那个”，系统报告当前画面未找到。
4. 切换目标丢失、画面冻结、运动无进展场景，观察任务失败及原因；运动期间可随时点击“停止”。

命令行支持多轮对话：

```powershell
python -X utf8 -m jarvis
python -X utf8 -m jarvis --once "帮我看看蓝色盒子"
python -X utf8 -m jarvis --scenario lost --once "蓝色盒子"
```

默认规则对话支持中文颜色、圆/方、盒子、左右、从左到右序号、否定和取消。它不是通用自然语言大模型。颜色检测限于红、蓝、绿、黄的清晰圆形/方形；请求已识别的其他颜色会得到未找到。任意物体或更自由表达应使用模型接口。

## 模型和语音接口

```powershell
python -m pip install -r requirements-optional.txt
$env:JARVIS_BASE_URL = "服务商提供的 OpenAI 兼容接口地址"
$env:JARVIS_API_KEY = "你的密钥"
$env:JARVIS_MODEL = "支持图像输入和 Function Calling 的模型名称"
python -X utf8 -m jarvis --model --detector vision --web
```

模型通过受限工具调用返回多候选、选择或澄清，不接收原始关节控制权。候选编号、目标框、动作种类都在本地校验。摄像头画面只有在显式启用实机时才会使用；上述命令给模型发送的是合成场景图。`--model --detector color` 则使用本地颜色检测、模型理解文字。

语音采用按回车开始、再次回车结束的录音，随后转录、执行同一寻物流程并朗读：

```powershell
$env:JARVIS_ASR_MODEL = "支持 audio/transcriptions 的转录模型名称"
python -X utf8 -m jarvis --model --voice --speak
```

这条命令会使用麦克风和外部接口，但运动仍为模拟。网页也可上传 WAV 录音。录音最多保留 30 秒；Ctrl+C 或网页停止入口在本地取消任务，不等待云端作出停止决定。已经发送的远程请求可能仍执行并计费，取消后返回的结果不会继续发起运动。按键语音不是随时唤醒或全双工打断。

没有预设供应商、型号、免费额度或价格。所选服务必须实际支持所用接口；本次只使用伪造响应验证接口处理，没有调用真实模型或测试麦克风/扬声器。密钥只从环境变量读取。请求超时为 30 秒且不自动重试。

## 完成条件和失败处理

- 正确目标中心进入画面中央宽、高各 20% 的区域，连续保持至少 1 秒。
- 使用另一张新画面重新检测并核对目标身份；模型返回后还会读取最新一帧，防止拿过期结果报告完成。
- 目标丢失、重复标记 ID、匹配不唯一、画面过期/重复、无进展、超时、运动异常或超出工作区，都会终止当前任务并记录失败。
- 本地跟踪保持初始模板，拒绝低置信度或多重匹配；它适合小幅连续视角变化，对明显缩放、遮挡、光照变化可能保守失败。最终物体标签不一致也会拒绝完成。
- 模型检测和身份核对本身存在误识别风险，软件检查通过不替代人工真值标注或实机验收。

## 运行记录与评测

每次启动创建 `runs/日期时间/`：

| 文件 | 内容 |
| --- | --- |
| `events.jsonl` | 用户指令、观察、澄清、每次运动请求、停止和核对事件 |
| `tasks.jsonl` | 每个完整任务的终态、耗时、澄清次数、目标、失败原因、API 统计 |
| `api_calls.jsonl` | 启用模型时每次接口调用的耗时、成功状态和服务商报告的 token 用量 |
| `*_before.jpg`、`*_after.jpg` | 执行前画面与成功后的最新画面；失败不生成成功后照片 |
| `*.wav` | 语音模式输入录音 |

首次响应时间记录到任务决策/追问；总耗时包含等待用户回答的时间。没有报告 token 时 `tokens_fully_reported=false`，累计数字仅为已知部分；费用 `null` 表示未知，以账单为准，不能当作免费。

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m jarvis.experiment --output runs/my_experiment
python -X utf8 -m jarvis.evaluate runs/my_experiment/trials.jsonl --output runs/my_summary.json
```

离线实验包括 20 个明确目标任务、10 组“直接选”与“先澄清”的配对任务、5 个不存在目标、3 个故障场景，共 48 次。每次重置对话，配对共享场景种子，固定种子随机安排执行顺序。实验采用虚拟时钟加速，澄清回答延迟人为设为 2 秒，不能据此估计真实交互耗时。直接选策略故意选择首个候选，只用于模拟对照，不在网页或实机开放。

汇总按模拟/实机、检测方法、策略分别统计，失败不从分母移除。`completed` 只表示程序完成了所选目标的居中；`selection_correct` 与 `task_correct` 才能判断是否满足场景真值。普通运行的真值未知，保持 `null`，现场评测需要人工标注。重试是新任务；命令行用 `--retry-of 原task_id` 标记首个任务为重试，汇总会单列。演示合成图像的成绩不能填入 proposal 的 20/10/5 项实机验收成绩。

## 后续实机接入边界

本次没有连接机械臂、相机或音频设备。`config/default.json` 中 `hardware.enabled=false`、`calibration.verified=false`，运动映射和工作区为空。即使添加 `--hardware` 也会拒绝启动。

后续接入需要根据当前装配实测确定串口、相机、关节映射、可达区，以及 `image_to_base_mm` 这个 3×2 矩阵：它把归一化图像误差 `[目标x-0.5, 目标y-0.5]` 映射为基座毫米位移，**正负方向不能猜测**。`workspace_mm` 为基座坐标三轴的已验证范围；默认单步上限 3 mm、一次启动累计上限 60 mm。模型坐标边界不构成独立碰撞检测。

`jarvis/hardware.py` 已接上本目录的 SCS215 底层实现，串口只在明确启用并通过配置检查后创建。失败或停止后锁存停止，网页重置不会重新解锁实机，必须结束程序后重新启动。旧课堂配置只是开发基础，不是当前装配已验收的证据。ArUco 模式通过 `--detector aruco` 选择，标记编号可直接作为目标；离线测试使用生成的标记图片检验检测和跟踪。

有限视角搜索和目标触碰是提案的条件扩展，本次不纳入主线完成声明：前者需要已经验证的观察姿态，后者还需要定位与接触精度实测。本版不会自动搜索视角、触碰、抓取或递物。

## 文件与来源

| 路径 | 职责 |
| --- | --- |
| `jarvis/controller.py` | 多轮任务状态、闭环居中、停止、最终核对与记录 |
| `jarvis/perception.py` | 颜色形状/ArUco 检测、规则澄清、本地图像跟踪 |
| `jarvis/providers.py`、`audio.py` | 模型工具调用、语音转录、录音和朗读 |
| `jarvis/simulation.py` | 合成场景与故障注入 |
| `jarvis/hardware.py` | 后续相机与机械臂适配 |
| `jarvis/server.py`、`web/index.html` | 本地网页与接口 |
| `jarvis/experiment.py`、`evaluate.py` | 配对实验、记录汇总 |
| `tests/` | 不访问真实模型或设备的回归测试 |
| `vendor/scs215/` | 原有课堂控制代码及必要配置 |

`vendor/scs215` 的 `advanced.py`、`servo_bus.py`、`joint_mapping.py`、`advanced.json`、`config.json`、`so101_new_calib.urdf` 来自工作区已有 `embodied-intelligence-course-week7/week4/so101_scs215_portable`。保留原作者与课堂工作归属；仅修改 `advanced.py` 两条导入以在本项目内运行。新增项目的硬件适配也保留其限位、逆解、动作等待、反馈错误处理。未修改原目录。

接口参考：[OpenAI Python 官方客户端](https://github.com/openai/openai-python)、[OpenCV ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)、[OpenCV 模板匹配](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)。
