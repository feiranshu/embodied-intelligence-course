import argparse
from datetime import datetime
import json
from pathlib import Path

from .controller import Assistant
from .perception import ArUcoDetector, ColorShapeDetector, LocalSelector
from .simulation import Simulation


ROOT = Path(__file__).resolve().parents[1]


def build(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    client = None
    if args.model:
        from .providers import ModelClient, ModelSelector, VisionDetector
        client = ModelClient()
        selector = ModelSelector(client)
        detector = VisionDetector(client) if args.detector == "vision" else None
    else:
        selector = LocalSelector()
        detector = None
        if args.detector == "vision":
            raise ValueError("视觉模型检测需要同时添加 --model")
    if detector is None:
        detector = ArUcoDetector() if args.detector == "aruco" else ColorShapeDetector()
    if args.hardware:
        if not config["hardware"]["enabled"]:
            raise ValueError("实机入口关闭：hardware.enabled=false。须完成现场标定后单独配置。")
        from .hardware import Camera, HardwareMotion
        motion = HardwareMotion(config["hardware"])
        try:
            camera = Camera(source=config["hardware"].get("camera", 0))
        except Exception:
            motion.close()
            raise
        mode = "hardware"
    else:
        camera = motion = Simulation(scenario=args.scenario)
        mode = "simulation"
    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        assistant = Assistant(camera, motion, detector, selector, config, run_dir, mode=mode)
        assistant.retry_of = args.retry_of
    except Exception:
        try:
            camera.close()
        finally:
            if motion is not camera:
                motion.close()
        raise
    return assistant, client


def main():
    parser = argparse.ArgumentParser(description="JARVIS 会追问的桌面寻物助手；默认只运行合成图像与模拟运动")
    parser.add_argument("--web", action="store_true", help="启动本地网页")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--once", help="执行一条文字指令")
    parser.add_argument("--retry-of", help="把本次首个任务标记为指定 task_id 的重试")
    parser.add_argument("--scenario", choices=Simulation.names, default="default")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    parser.add_argument("--detector", choices=["color", "aruco", "vision"], default="color")
    parser.add_argument("--model", action="store_true", help="使用配置的真实模型接口（可能计费）")
    parser.add_argument("--voice", action="store_true", help="回车开始/结束录音，需要模型转录接口")
    parser.add_argument("--speak", action="store_true", help="本地朗读回复")
    parser.add_argument("--hardware", action="store_true", help="显式启用经验证的实机配置")
    args = parser.parse_args()
    if args.voice and not args.model:
        parser.error("--voice 需要 --model 和已配置的 JARVIS_ASR_MODEL")
    if args.web and (args.voice or args.once):
        parser.error("--web 不与 --voice/--once 同时使用")
    if args.detector == "aruco" and not args.hardware:
        parser.error("默认合成场景没有标记；ArUco 可由测试图像验证，现场使用需配置实机相机")
    assistant = None
    try:
        assistant, client = build(args)
        print("模式：" + ("软件模拟（不连接串口、相机或麦克风）" if not args.hardware and not args.voice
                         else "实机" if args.hardware else "模拟运动＋真实语音输入"))
        print(f"运行记录：{assistant.run_dir}")
        if args.web:
            from .server import serve
            serve(assistant, port=args.port, model_client=client)
            return
        print("示例：帮我看看蓝色盒子 / 红色那个 → 圆的 / 紫色物体。Ctrl+C 本地停止；q 退出。")
        while True:
            wav = None
            if args.once is not None:
                text = args.once
            elif args.voice:
                typed = input("回车开始录音，q 退出：").strip()
                if typed.lower() == "q":
                    break
                from .audio import record_push_to_talk
                wav = record_push_to_talk(start_prompt=False)
                text = ""
            else:
                text = input("你：").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text and wav is None:
                continue
            state = assistant.run(text, wav=wav, model_client=client)
            print("JARVIS：" + state["reply"])
            if args.speak:
                from .audio import speak
                speak(state["reply"])
            if args.once is not None:
                break
    except KeyboardInterrupt:
        print("\n已请求本地停止。")
    except Exception as error:
        parser.exit(1, f"启动或运行失败：{error}\n")
    finally:
        if assistant:
            assistant.close()


if __name__ == "__main__":
    main()
