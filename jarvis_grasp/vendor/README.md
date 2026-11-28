# SCS215 控制代码来源

本目录仅复制 `embodied-intelligence-course-week7/week4/so101_scs215_portable` 中的六个运行依赖：`advanced.py`、`servo_bus.py`、`joint_mapping.py`、`advanced.json`、`config.json`、`so101_new_calib.urdf`。`advanced.py` 的两条导入改为包内相对导入；其他控制逻辑及配置保留原样。

旧配置中的标定标志仅表示历史记录，不能证明当前装配已验证。JARVIS 在打开串口前另外要求显式启用硬件、当前图像运动标定 `verified=true`、有效矩阵和已验证工作区。默认项目配置不满足这些条件。

这里的运动学坐标属于模型估计，不是外部测量。工作区检查不能替代机械臂全连杆碰撞检查。尚未进行本项目的实机验证。
