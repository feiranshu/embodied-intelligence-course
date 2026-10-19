# 具身智能课程作业

SO-101 结构、六只 Feetech SCS215 舵机的课程实验、控制程序和演示材料。

接手项目先看 [交接说明](HANDOFF.md)，其中列出了安装方法、验证结果、已知问题和后续验收事项。

在新电脑上操作时先进入**本 README 所在目录**；如果下载的压缩包外层还有一个同名文件夹，以包含本文件、`so101_scs215_portable/` 和 `so101_scs215_position_homework/` 的目录为项目根目录。具体步骤见 [本地复现说明](本地复现.md)。

| 内容 | 入口 |
|---|---|
| 新电脑安装与离线复现 | [本地复现.md](本地复现.md) |
| 全部文件的用途 | [文件清单.md](文件清单.md) |
| 后续进阶工程：预设姿态/动作、URDF IK、中文语音、视觉实验 | [so101_scs215_portable](so101_scs215_portable/README.md)、[进阶使用说明](so101_scs215_portable/进阶使用说明.md) |
| 原有六轴位置模式作业 | [so101_scs215_position_homework](so101_scs215_position_homework/README.md) |
| 第六周视觉伺服四部分作业 | [week6_visual_servo 操作说明](week6_visual_servo/README.md) |
| Hackathon 任务报告 | [Hackathon任务报告.md](Hackathon任务报告.md) |
| 双舵机课堂实验 | [课堂练习报告](课堂练习报告.md)、[classroom_servo.py](classroom_servo.py)、[serial_port.py](serial_port.py) |
| LeRobot 功能梳理 | [林邵坪24307090026.md](林邵坪24307090026.md) |
| 实机演示视频 | [立正](演示视频/立正.mp4)、[动作2](演示视频/动作2.mp4)、[动作3](演示视频/动作3.mp4)、[语音控制](演示视频/语音控制.mp4) |

进阶工程包含 Vosk 中文模型和离线测试录音。源码、报告与视频按原目录结构保存，不需要本机原有的虚拟环境或 LeRobot 源码目录。

**状态说明：** 1.1 预设姿态和 2.1 语音已有历史实机演示；1.2 已实现且旧说明登记为完成，但完整六方向、定距及外部测量证据尚缺；2.2 视觉闭环尚未完成。具体证据与限制见交接说明。
