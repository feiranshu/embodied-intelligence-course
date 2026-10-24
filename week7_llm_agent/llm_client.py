"""OpenAI-compatible clients for the course's LLM platforms.

API keys are read only from environment variables, never from files in the
repository.  Every provider can be overridden with --base-url / --model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_env: tuple[str, ...]
    vision_model: str
    agent_model: str
    # Extra body that turns off "thinking" so replies come back quickly.
    no_thinking: dict = field(default_factory=dict)


PROVIDERS = {
    # 智谱：GLM-4.6V-Flash 免费，支持图像理解与工具调用。
    "zhipu": Provider("https://open.bigmodel.cn/api/paas/v4/", ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY"),
                      "glm-4.6v-flash", "glm-4.6v-flash", {"thinking": {"type": "disabled"}}),
    # 硅基流动：Qwen/Qwen3.5-4B 免费。
    "siliconflow": Provider("https://api.siliconflow.cn/v1", ("SILICONFLOW_API_KEY",),
                            "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B", {"enable_thinking": False}),
    # 阿里云百炼：全模态 qwen3.8-omni-flash 支持图像＋音频输入和 Function Calling。
    "dashscope": Provider(os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                          ("DASHSCOPE_API_KEY",), "qwen3.8-omni-flash", "qwen3.8-omni-flash",
                          {"reasoning_effort": "none"}),
    # 已在 week4 使用过的 DeepSeek，仅文本，可用于 4.2 的纯文本工具调用。
    "deepseek": Provider("https://api.deepseek.com", ("DEEPSEEK_API_KEY",), "", "deepseek-chat", {}),
}


@dataclass
class LLM:
    client: object
    model: str
    extra_body: dict
    provider: str

    def chat(self, messages, **kwargs):
        """Call chat.completions, retrying only transient provider/network failures."""
        extra = {**self.extra_body, **kwargs.pop("extra_body", {})}
        if extra:
            kwargs["extra_body"] = extra
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        transient = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        delays = (0, 3, 8)
        for attempt, delay in enumerate(delays, 1):
            if delay:
                print(f"模型服务暂时繁忙，第 {attempt}/{len(delays)} 次尝试将在 {delay} 秒后进行…", flush=True)
                time.sleep(delay)
            try:
                return self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
            except transient:
                if attempt == len(delays):
                    raise


def add_llm_arguments(parser, *, default_provider="zhipu", purpose="vision"):
    group = parser.add_argument_group("LLM 接口")
    group.add_argument("--provider", choices=sorted(PROVIDERS), default=default_provider,
                       help=f"模型平台，默认 {default_provider}")
    group.add_argument("--model", help="模型名；缺省使用平台预设")
    group.add_argument("--base-url", help="覆盖平台预设的 OpenAI 兼容地址")
    group.add_argument("--api-key-env", help="从指定环境变量读取 API Key")
    group.add_argument("--thinking", action="store_true", help="保留模型思考（默认关闭以加快响应）")
    parser.set_defaults(llm_purpose=purpose)
    return group


def make_llm(provider="zhipu", *, model=None, base_url=None, api_key_env=None,
             purpose="vision", thinking=False, client=None) -> LLM:
    preset = PROVIDERS[provider]
    model = model or (preset.vision_model if purpose == "vision" else preset.agent_model)
    if not model:
        raise ValueError(f"{provider} 没有预设的{('视觉' if purpose == 'vision' else '对话')}模型，请用 --model 指定")
    if client is None:
        names = (api_key_env,) if api_key_env else preset.key_env
        key = next((os.environ[name] for name in names if os.environ.get(name)), None)
        if not key:
            raise RuntimeError(f"请先设置环境变量 {' 或 '.join(names)}（API Key 不写进代码或仓库）")
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url or preset.base_url, timeout=60)
    return LLM(client, model, {} if thinking else dict(preset.no_thinking), provider)


def llm_from_args(args, client=None) -> LLM:
    return make_llm(args.provider, model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                    purpose=args.llm_purpose, thinking=args.thinking, client=client)
