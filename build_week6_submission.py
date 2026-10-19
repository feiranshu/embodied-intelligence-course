"""Assemble the Week 6 submission from current source and measured data."""
from pathlib import Path
import json
import shutil
import html
import sys
import zipfile
import hashlib

ROOT=Path(__file__).resolve().parent
OUTER=ROOT.parent
NAME='23-束斐然-方震宇-林邵坪'
DEST=OUTER/'第6周提交材料'/NAME
W=ROOT/'week6_visual_servo'
ARM=ROOT/'week4/so101_scs215_portable'


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def launcher(part):
    return f'''@echo off
setlocal
title Week 6 - Part {part}
set "TASK_DIR=%~dp0代码\\week6_visual_servo\\part{part}_{'hand_eye_calibration' if part==2 else 'visual_servo'}"
set "TASK_PYTHON="
if defined CONDA_PREFIX if /i "%CONDA_DEFAULT_ENV%"=="py312" if exist "%CONDA_PREFIX%\\python.exe" set "TASK_PYTHON=%CONDA_PREFIX%\\python.exe"
if not defined TASK_PYTHON if exist "D:\\software\\anaconda\\envs\\py312\\python.exe" set "TASK_PYTHON=D:\\software\\anaconda\\envs\\py312\\python.exe"
if not defined TASK_PYTHON if exist "%USERPROFILE%\\anaconda3\\envs\\py312\\python.exe" set "TASK_PYTHON=%USERPROFILE%\\anaconda3\\envs\\py312\\python.exe"
if not defined TASK_PYTHON if exist "%USERPROFILE%\\miniconda3\\envs\\py312\\python.exe" set "TASK_PYTHON=%USERPROFILE%\\miniconda3\\envs\\py312\\python.exe"
if not defined TASK_PYTHON (
 echo Activate Conda py312 in PowerShell and run the Python command in README.md.
 pause
 exit /b 1
)
cd /d "%TASK_DIR%"
echo Close other webcam and servo services first.
echo Opening the page does not start robot motion.
"%TASK_PYTHON%" -X utf8 "%TASK_DIR%\\web_app.py"
pause
'''


def result_page(title,body):
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+title+'''</title><style>body{margin:0;background:#eef2f7;color:#172638;font:16px/1.6 "Microsoft YaHei",sans-serif}main{max-width:1000px;margin:22px auto;padding:24px 32px;background:white;border-radius:12px}h1{font-size:27px;margin:0 0 8px}h2{font-size:19px;margin:14px 0 5px}pre{background:#f2f5f9;padding:14px;font:17px/1.65 Consolas,monospace;white-space:pre-wrap;border-radius:7px}b{color:#175cba}.sub{color:#58677c}.metric{font-size:22px;padding:12px;background:#edf5ff;border-left:5px solid #2465db}p{margin:8px 0}</style><main><h1>'''+title+'''</h1><div class="sub">第23组 · 束斐然、方震宇、林邵坪 · 实验数据结果展示</div>'''+body+'''</main></html>'''


def build():
    if DEST.exists():raise FileExistsError('Submission folder already exists; preserve it and inspect before rebuilding')
    code=DEST/'代码';wc=code/'week6_visual_servo';ac=code/'week4/so101_scs215_portable'
    DEST.mkdir(parents=True)
    for name in ['common.py','camera_utils.py','board_orientation.py','check_camera.py','requirements.txt','test_offline.py','test_board_orientation.py']:
        copy_file(W/name,wc/name)
    for part in ['part1_camera_calibration','part3_visual_servo','part4_object_tracking']:
        shutil.copytree(W/part,wc/part,ignore=shutil.ignore_patterns('__pycache__','runs','*.pyc'))
    p2=wc/'part2_hand_eye_calibration'
    for name in ['hand_eye.py','web_app.py','web_ui.html','record_poses.py','pose_plan.example.json','pose_plan.json',
                 'verify_final_calibration.py','finalize_calibration.py','test_web_ui.mjs']:
        copy_file(W/'part2_hand_eye_calibration'/name,p2/name)
    final=W/'part2_hand_eye_calibration/最终标定结果'
    shutil.copytree(final,p2/'最终标定结果')
    for name in ['hand_eye.json','hand_eye.png']:copy_file(final/name,p2/name)
    samples=json.loads((final/'inputs/samples.json').read_text(encoding='utf-8'))
    write(p2/'samples/samples.json',json.dumps(samples,ensure_ascii=False,indent=2))
    for sample in samples['samples']:
        copy_file(final/'inputs/images'/Path(sample['image']).name,p2/'samples'/sample['image'])
    # Preserve dependencies next to week6; no environment, models or servo recordings.
    shutil.copytree(ARM,ac,ignore=shutil.ignore_patterns('__pycache__','*.pyc','models','rotation_records','verification','.git','AGENTS.md','HANDOFF.md'))
    for index in range(1,21):
        copy_file(W/f'part1_camera_calibration/evidence/task1_handoff/intrinsic_{index:02}.png',
                  wc/f'part1_camera_calibration/images/chess_{index:02}.png')
    write(wc/'part1_camera_calibration/images/board.json',json.dumps(dict(inner_corners=[8,6],square_mm=25)))
    intr=json.loads((W/'part1_camera_calibration/camera_intrinsics.json').read_text(encoding='utf-8'))
    hand=json.loads((final/'hand_eye.json').read_text(encoding='utf-8'))
    for src,name in [(W/'part1_camera_calibration/evidence/task1_handoff/角点检测与重投影总览.jpg','任务1_角点与重投影误差.jpg'),
                     (W/'part1_camera_calibration/evidence/task1_handoff/去畸变对比.jpg','任务1_去畸变对比.jpg'),
                     (final/'hand_eye.png','任务2_程序输出结果.png')]:copy_file(src,DEST/'截图'/name)
    K=intr['camera_matrix'];D=intr['dist_coeffs'];X=hand['T_gripper_camera']
    matrix=lambda value:'\n'.join('['+'  '.join(f'{x:12.6f}' for x in row)+']' for row in value)
    write(DEST/'结果页面/相机标定结果.html',result_page('任务一：相机标定运行结果',
        f'<p>GENERAL WEBCAM · 640×480 · 20张实拍图像 · 8×6内角点 · 方格25 mm</p><div class="metric">整体重投影 RMS：<b>{intr["rms_px"]:.6f} px</b></div><h2>相机内参 K</h2><pre>{matrix(K)}</pre><h2>畸变系数</h2><pre>{", ".join(f"{v:.8f}" for v in D)}</pre><p class="sub">数值读取本组相机标定结果JSON；原始图像、角点总览与复算程序随包提供。</p>'))
    write(DEST/'结果页面/手眼标定结果.html',result_page('任务二：手眼标定运行结果',
        f'<p>基于本组实机采集的12组图像与舵机反馈求得以下标定矩阵。</p><h2>T_gripper_camera（平移单位 mm）</h2><pre>{matrix(X)}</pre><div class="metric">全链重投影 RMS：<b>{hand["rms_px_selected"]:.6f} px</b></div><p>棋盘位置散布RMS：{hand["board_origin_scatter_rms_mm"]:.2f} mm；方向散布RMS：{hand["board_orientation_scatter_rms_deg"]:.2f}°。</p><p>含义：p<sub>末端</sub> = R · p<sub>相机</sub> + t。相机坐标转换为URDF末端坐标。</p><p class="sub">散布表示样本一致性，不能代替绝对定位误差。关节换算与矩阵配套保存；实机定位精度未独立验收。</p>'))
    write(DEST/'视频/待补视频说明.md','''# 演示视频待补

当前尚未拍摄演示视频，不能将本包视为已满足全部提交项。

请补入 `任务3_视觉伺服演示.mp4`：同一段真实视频同时展示机械臂实际运动和网页识别画面，标记先偏离中央，点击持续调整，最后展示到位或真实停止状态。建议保留原声说明，不要把离线模拟剪成实机演示。

拍摄完成后将文件放在本目录，在包根目录运行“补充视频并重新打包.ps1”，生成同名ZIP。教师要求10月18日24:00前提交。
''')
    report=f'''# 第6周实验报告：相机标定、手眼标定与简单视觉伺服

**组号：23**  
**组员：束斐然、方震宇、林邵坪**  
**设备：六舵机SCS215机械臂、GENERAL WEBCAM**

## 一、实验内容与文件

完成相机标定、基于本组实机采集数据的手眼标定，以及ArUco简单视觉伺服程序。代码、原始数据、结果和运行说明随本报告提交。**视觉伺服演示视频尚未拍摄，待补入视频目录。**

## 二、相机标定

使用20张640×480实拍图像，棋盘为8×6内角点、方格边长25 mm。程序提取角点，调用OpenCV相机标定，得到内参及畸变系数；用重投影RMS评价像素误差。

```text
K =
{matrix(K)}

畸变系数 [k1,k2,p1,p2,k3] =
{[round(x,8) for x in D]}
```

整体重投影RMS为 **{intr['rms_px']:.6f} px（约0.498 px）**。

![相机标定程序结果页面](截图/任务1_运行截图.jpg)

![20张实拍图像的角点与重投影误差](截图/任务1_角点与重投影误差.jpg)

图中包含整体重投影误差和逐图误差。原始图像保存在代码中的 `part1_camera_calibration/images/`，程序结果保存在 `camera_intrinsics.json`。

## 三、手眼标定

### 3.1 数据采集与求解

相机刚性固定于机械臂末端，棋盘固定。本组实机采集12组图像和对应的六轴舵机反馈，使用蓝圈统一棋盘角点原点。程序由图像估计棋盘相对相机的位姿，并由关节读数计算末端位姿，建立关系：

```text
T_base_board = T_base_gripper × T_gripper_camera × T_camera_board
```

以固定棋盘在基座坐标系中的一致性和角点投影为依据，求得本组最终标定矩阵。相机坐标采用OpenCV约定；平移单位为毫米。矩阵与逐关节方向、角度换算参数配套保存在JSON中。

### 3.2 本组实测数据得到的最终标定矩阵

```text
T_gripper_camera =
{matrix(X)}
```

![手眼标定程序结果页面](截图/任务2_运行截图.jpg)

### 3.3 矩阵含义

左上3×3矩阵R描述相机坐标轴相对末端坐标轴的方向，右上3×1向量t描述相机光学原点在末端坐标系中的位置：

```text
p_gripper = R × p_camera + t
t = [28.438583, 150.041583, -343.733152] mm
```

末端坐标为URDF的 `gripper_frame_link`；其原点由模型定义，不能默认等同于夹爪尖端。t的长度约376.13 mm，是该解的输出，尚未通过原点明确的外部测量确认。

### 3.4 结果与误差解释

| 指标 | 结果 |
| --- | ---: |
| 数据组数 | 12组，全部使用 |
| 全链角点重投影RMS | {hand['rms_px_selected']:.6f} px |
| 棋盘位置散布RMS | {hand['board_origin_scatter_rms_mm']:.2f} mm |
| 棋盘方向散布RMS | {hand['board_orientation_scatter_rms_deg']:.2f}° |

位置与方向散布衡量本批样本一致性，不能作为绝对定位误差或“小于50 mm”的实测证明。ID1、ID2在该轮变化较小，不能由这批数据独立确认其角度换算。当前结果尚未完成独立实机定位精度验收。

## 四、简单视觉伺服实现思路

1. 用ArUco `DICT_4X4_50` 字典识别指定ID，默认ID0，计算标记中心；已知边长时估计相机Z方向深度。
2. 默认以画面横纵40%～60%的中央黄色框为目标区域，可修改区域比例。
3. 标记偏左、右、上、下时，按偏差更大的方向请求一次末端位移，默认步长10 mm。
4. 用当前末端朝向与手眼矩阵将相机方向转换到机械臂基座，调用配套FK/IK和反馈控制。
5. 动作后读取新画面，重复识别与调整；标记中心进入中央区域即完成。默认只居中，不向前靠近。
6. 普通识别失败时尝试对比度增强和2倍放大识别；短暂漏检等待重新识别，不以旧图像发出新运动。
7. 网页提供单步、持续调整和停止按钮。通信异常、画面中断、停止请求或控制器无法完成动作时结束，并保存真实状态日志。

每次10 mm是模型中的末端位移请求，不是外部测量确认的真实相机移动距离。当前已完成程序实现和离线检查；完整实机闭环效果以待补演示视频为准。

**演示视频：待补 `视频/任务3_视觉伺服演示.mp4`。**

## 五、week4相关修正与检查

将逐关节带方向角度比例、参考读数和参考角统一接入week4运动学核心，使FK、IK边界和读数反算使用相同参数。任务三复用该接口，不改写原advanced.json。参考角变化时同步变换URDF角度边界，保持原允许的物理读数集合。

SCS215在读数保持不变的区间不能仅靠反馈恢复真实角度；本次修正没有声称解决传感器未提供信息的问题。原URDF正运动学矩阵乘法未发现需要整体改写的错误。原机械臂范围、跨零处理、故障反馈和停止检查保留。

离线验证包括：week6的27项识别、坐标、任务逻辑与清理检查，以及week4逐关节换算与原有运动学等8项相关检查。它们不代替机械臂现场验收。启动方式、复算方式和视频补交步骤见根目录README。
'''
    write(DEST/'报告.md',report)
    readme=r'''# 第6周提交材料 · 第23组

组员：**束斐然、方震宇、林邵坪**。提交截止：**10月18日24:00**。

## 当前提交状态

- 代码、Markdown报告、相机／手眼结果与截图、原始标定数据已整理。
- **演示视频待补**，当前ZIP是视频待补版，尚未满足完整提交要求。
- `报告.md` 是正式实验报告，`截图/` 是结果截图与原始输出图。
- `代码/week6_visual_servo/` 含四部分代码；`代码/week4/so101_scs215_portable/` 是本机控制依赖及修正。
- `结果页面/` 是由已有实验结果生成的可离线查看页面，截图来自该页面；不是对尚未拍摄的实机演示的替代。

## 启动网页

先关闭其他占用摄像头或串口的程序。双击根目录“启动任务二网页.cmd”或“启动任务三网页.cmd”。启动只预览，不自动运动。任务三默认持续居中，标记中心进入黄框即完成；实机前先单步检查方向。

启动器使用已安装Conda py312；找不到环境时，在本目录PowerShell运行：

```powershell
conda activate py312
python -m pip install -r .\代码\week6_visual_servo\requirements.txt
python .\代码\week6_visual_servo\part3_visual_servo\web_app.py
```

任务二改为 `python .\代码\week6_visual_servo\part2_hand_eye_calibration\web_app.py`。使用不同电脑时无需修改原项目绝对路径；设备名或端口不同可通过命令参数指定。

## 复核已保存结果

```powershell
conda activate py312
Set-Location .\代码\week6_visual_servo
python .\part2_hand_eye_calibration\verify_final_calibration.py
python -m unittest part3_visual_servo.test_simple_servo test_offline test_board_orientation -q
```

相机原始20张图像已整理为默认的 `images/chess_*.png`，可重新计算到另一个文件，保留本组已提交内参：

```powershell
python .\part1_camera_calibration\camera_calibration.py calibrate --cols 8 --rows 6 --square-mm 25 --output .\part1_camera_calibration\camera_intrinsics_recomputed.json
```

第三部分默认读取 `part2_hand_eye_calibration/最终标定结果/hand_eye.json` 及配套关节换算，不应只复制矩阵搭配旧角度模型。程序与标定的局限详见报告，不能把模型位移当外部测量。

## 补视频后重新打包

1. 把真实演示视频命名为 `任务3_视觉伺服演示.mp4`，放进 `视频/`。
2. 更新报告中“视频待补”为实际完成情况，保留真实失败／停止状态，不冒充成功。
3. 回到包根目录，在PowerShell运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\补充视频并重新打包.ps1
```

脚本会重新生成父目录的同名ZIP，包含新增视频；不会删除原目录。提交同名ZIP即可。
'''
    write(DEST/'README.md',readme)
    for part in [2,3]:
        # UTF-8 code page is set before the first Chinese path is parsed.
        text=launcher(part).replace('@echo off','@echo off\nchcp 65001 >nul',1)
        write(DEST/f'启动任务{"二" if part==2 else "三"}网页.cmd',text)
    repack='''$ErrorActionPreference = 'Stop'
$packageName = Split-Path -Leaf $PSScriptRoot
$destinationZip = Join-Path (Split-Path -Parent $PSScriptRoot) ($packageName + '.zip')
$videoDirectory = Join-Path $PSScriptRoot '视频'
$videos = @(Get-ChildItem -LiteralPath $videoDirectory -File | Where-Object { $_.Extension.ToLowerInvariant() -in @('.mp4','.mov','.mkv','.avi') })
if ($videos.Count -eq 0) { Write-Warning '视频仍未补齐：本次压缩包还不是完整提交版。' }
Compress-Archive -LiteralPath $PSScriptRoot -DestinationPath $destinationZip -Force
Write-Host ('已生成：' + $destinationZip)
'''
    (DEST/'补充视频并重新打包.ps1').write_text(repack,encoding='utf-8-sig')
    write(wc/'README.md','''# week6程序运行

请优先阅读提交包根目录README和报告。任务三详细操作见 `part3_visual_servo/README.md`，任务二结果见 `part2_hand_eye_calibration/最终标定结果/README.md`。相机棋盘规格为8×6内角点、方格25 mm。
''')
    print(DEST)


def finish():
    for name in ['任务1_运行截图.jpg','任务2_运行截图.jpg']:
        if not (DEST/'截图'/name).is_file():raise FileNotFoundError('Capture result page first: '+name)
    files=[]
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and p.name!='打包记录.json' and '__pycache__' not in p.parts and p.suffix!='.pyc':
            files.append(dict(path=p.relative_to(DEST).as_posix(),size=p.stat().st_size,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write(DEST/'打包记录.json',json.dumps(dict(group=23,members=['束斐然','方震宇','林邵坪'],
        video_pending=not any((DEST/'视频').glob('*.mp4')),files=files),ensure_ascii=False,indent=2))
    output=DEST.parent/(NAME+'.zip')
    temporary=output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':z.write(p,Path(NAME)/p.relative_to(DEST))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP integrity check failed')
    temporary.replace(output)
    print(output,output.stat().st_size)


if __name__=='__main__':
    if '--finish' in sys.argv:finish()
    else:build()
