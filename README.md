# manju

AI 漫剧生成 pipeline — 打工牛马 / 健身题材幽默向竖屏短视频。

姊妹项目:`../fitness-video-pipeline`(处理式视频流水线)。本项目**生成式**,物理隔离不共享代码。

## 快速开始
```bash
uv sync                  # 建独立 venv
# uv run python ...      # (后续 pipeline 脚本)
```

## 工作流(7 阶段,可插拔 provider)
```
剧本(LLM) → 生图(ComfyUI/火山) → 【角色一致性·命门】
        → 图生视频(可灵/火山) → 配音(火山TTS/MiniMax)
        → 后处理(ffmpeg:角标/字幕/调色) → 导出(竖版三件套)
```
详见 `CLAUDE.md`。

## 平台 API 可行性(2026)
- **即梦**:网页无 API,但底层 Seedream/Seedance 经**火山引擎方舟**开放 API
- **可灵**:有官方商业 API(图生视频强)
- **ComfyUI**(本地):自带 HTTP API,零外部成本,最适合生图自动化
- **剪映**:无 API(用火山 TTS 替代其配音)
- **MiniMax / ElevenLabs**:配音 API

策略:MVP 先手动验证(零成本),过了再上 API 量产。

## 当前进度
- ep01《一个举过铁的牛马》:剧本 + 即梦提示词就绪
- 下一步:即梦手动生主角立绘,验证角色一致性
