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

## Wan-Animate 动作克隆 (当前主线, 2026-08-11 验证)

动作克隆 = 给一张角色参考照 + 一段驱动视频(动作), 生成该角色做该动作的新视频。
本地 ComfyUI + Wan2.2-Animate-14B (Q4 GGUF, RTX 4070 12GB DynamicVRAM, ~15min/25步)。

### 端到端链 (一键)
```bash
PY=/f/wkspace/fitness-video-pipeline/.venv/Scripts/python.exe   # face_protect 要 insightface

# 1. 生成 (ComfyUI 起着; canvas 必须等于驱动源尺寸, 否则鬼影)
$PY scripts/gen_wan_animate.py --width 768 --height 1152 \
    --drive drive_dance_768x1152.mp4 --coach client_a --zoom 1.0 --vshift 0
#   → output/wan_animate/wan_client_a_768x1152_matchsrc_00001_.mp4  (raw 768)

# 2. 后处理一键链: trim(裁渗出影子) → Real-ESRGAN(身2×锐) → face_protect(贴回好脸)
$PY scripts/build_wan_animate_demo.py \
    --input output/wan_animate/wan_client_a_768x1152_matchsrc_00001_.mp4 --coach client_a
#   → output/wan_animate/client_a_1536x2304_matchsrc_t9_faceprot.mp4  (最终交付, 1536×2304)
```

### 钉死规则 (别随手改, 详见 `memory/`)
- **canvas = 驱动源尺寸**: 生成画布必须 = 驱动视频分辨率 (768 驱动 + 768 canvas 一致), 否则鬼影/重影。要更近用 `--zoom`(裁驱动), 不能在生成阶段放大画布。
- **裁首 9 帧非 2**: 满 h_frac≈0.96 四视图参考照, 开头 f0–f8 左上角有参考照渗出影子 (缩略图可见); `--frames 9` 才无影。半身参考照 (h_frac~0.50) 渗出轻, 2 帧够。
- **脸是生成端解决的**: 768 全身脸 58px (vs 448 的 34px), 原始生成即可看。**别超分脸 / 别 GFPGAN** (僵尸脸/鬼脸)。Real-ESRGAN 只动身/背景, 脸由 face_protect 贴回原始。
- **清晰度差距是生成端结构问题**: 源真视频比生成锐 11–22×, 后处理封顶 2×。用户拍板交付 1536 不再追; 要更清需重生成 1024 (canvas=driver 一致技巧, 待 ComfyUI 起 + 验鬼影)。
- **换脸不值得磨**: 原始生成脸够好, 不上 facelock/inswapper。

### 关键脚本
| 脚本 | 作用 |
|------|------|
| `scripts/gen_wan_animate.py` | ComfyUI 提交 Wan-Animate 生成 (参数化 + 进度轮询) |
| `scripts/build_wan_animate_demo.py` | **一键后处理链** (trim → upscale → face_protect, 幂等) |
| `scripts/trim_startup.py` | 裁开头 N 帧 (特写 + 渗出影子) |
| `scripts/upscale_realesrgan.py` | Real-ESRGAN 超分 (animevideov3, 视频/真人脸自然) |
| `scripts/face_protect_upscale.py` | 贴回原始好脸 (治超分僵尸脸) |

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
