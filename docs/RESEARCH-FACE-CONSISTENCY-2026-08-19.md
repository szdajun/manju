# 调研报告: i2v 场景下人脸跨镜一致性解决方案 (2026-08-19)

**背景**: manju 项目用 MiniMax Hailuo H3 (ComfyUI 本地 GGUF 量化, 12GB 显卡) 做图生视频, 关键帧由 SDXL + IPAdapter (face/body/scene 三 adapter) 生成。痛点: i2v 模型重渲人脸, 同一角色跨镜 ArcFace 余弦相似度仅 ~0.28 (健康线 0.66–0.70, ArcFace 验证通行阈值 0.3–0.6, 参考型应用建议 ≥0.6), 角色每镜"长得不太一样"。

**参考基线**: ArcFace/InsightFace 余弦相似度 — 0.3–0.5 是学术上"同一人"的常见验证阈值区间, 0.6 是实践中较严格的同阈值 (ResearchGate 验证研究最佳阈值 0.6; StyleID 论文采用 0.3–0.5 区间)。我们的 0.28 连"勉强同一人"线都没过。

---

## 附 0: 仓库内资产盘点 (controller 补, 2026-08-19) — 上一轮换脸取消的档案复盘

2026-08-15 曾用 `scripts/facelock_h3_ep.py` (inswapper) 换脸, 指标成功 (跨镜 0.28→0.88) 但用户看片否决, 三坑已定位 (HANDOFF 记录)。本轮盘点确认**三坑全部可修, 且修复成本递减**:

| 坑 | 根因 | 修复 |
|---|---|---|
| ①音频丢失 | `facelock` 用 `cv2.VideoWriter` 只写视频流 (facelock_wan_animate.py:184-185); 而 **H3 i2v 产物自带 aac 音频流** (对白烧在里面, ffprobe 已确认 ep20_shot01.mp4 = h264+aac) | ffmpeg remux 一行: `ffmpeg -i 换脸视频 -i 原视频 -map 0:v -map 1:a -c copy`; 或直接改脚本在 facelock 末尾加 remux |
| ②换错脸 (男人脸换女人身) | `facelock` 逐帧取**最大脸**做目标; 双人/误检时换错 ref | 源 ref 已按 speaker 选取 (脚本:50-52); 目标脸改用 insightface 嵌入比对 face_ref 选**最近脸** (非最大脸), ~20 行 |
| ③嘴唇深/适配差 | `skin_and_lip_fix` 参数 (lip_desat=0.35/lum_blend=0.6) 为 Wan-Animate 输出调校, H3 渲染特性不同 | 参数暴露成 CLI (脚本已有 --color-match 先例), 对 H3 产物重调一轮即可 |

**本地资产现状**: inswapper_128.onnx (554MB, ~/.insightface/models/) + fitness venv insightface ✓ 在位; PuLID 未装; IPAdapter-plus-face SDXL ✓ 在位; ReActor/FaceFusion 未装。

**结论**: 方向 1 (后处理换脸) 我们不是从零开始 — 脚本/模型/后处理链全在, 修三坑 (~半天) vs 引入 FaceFusion (装新工具+适配)。见推荐排序修正。

---

## 方向 1: 视频后处理换脸 (i2v 之后统一"钉"脸)

### 1.1 FaceFusion (roop 的活跃后继, 推荐)
- **原理**: inswapper_128 (InsightFace) 逐帧换脸 + CodeFormer/GFPGAN 面部修复; CLI 驱动, 天然适合脚本化管线。
- **效果**: 换脸本身把身份钉死到参考脸 — 相似度不再依赖 i2v 质量, ArcFace 相似度可回到 0.8+ (同源 inswapper 的身份嵌入就是 ArcFace 系)。inswapper_128 仍是视频换脸事实标准 (r/FaceFusion 2025 讨论确认)。
- **音频**: **默认保留目标视频音轨**, `--skip-audio` 才会去掉; 还有 `--output-audio-volume/quality/encoder` 可控 (新版默认 flac)。无需手动 remux。
- **选说话人而非最大脸**: `face_selector_mode = reference` + 参考脸图 + `reference_face_distance` (默认 0.6, 角度大/遮挡时调高)。这是官方文档明确支持的用法, 正是"按身份选目标脸"的正解。
- **成本**: inswapper_128 是 ONNX CPU/GPU 均可, 显存 <2GB; 1080p 视频处理速度约实时 0.1–0.5x。
- **风险**: inswapper 输出面部分辨率硬上限 **128×128**, 特写大脸会糊/塑料感 → 必须叠 CodeFormer (v1.4 更脆), 而修复模型又会抹掉一些表情细节; 侧脸 >50% 遮挡时崩; 表情极端 (咧嘴/低头) 时贴合变形。
- 来源: [FaceFusion face-selector 官方文档](https://docs.facefusion.io/usage/cli-arguments/face-selector) | [output-creation 文档](https://docs.facefusion.io/usage/cli-arguments/output-creation) | [LoopIn 使用指南](https://docs.loopin.network/tutorials/faceswap/facefusion-usage) | [r/FaceFusion inswapper 现状讨论](https://www.reddit.com/r/FaceFusion/comments/1idoesk/is_inswapper_128_still_the_best_model_for_video/)

### 1.2 ReActor (ComfyUI 节点内, 管线内一体化)
- **原理**: Gourieff 的 ComfyUI 换脸节点, 同样吃 inswapper_128.onnx, 节点内集成 CodeFormer/GFPGAN 修复, 配 Video Helper Suite (VHS) 逐帧处理后重组视频。
- **效果**: 同 inswapper 上限; 社区口碑组合 = inswapper fp16 + CodeFormer 1.4 (+ 可选 RIFE 补帧)。有现成 CivitAI 免费工作流 "face swap onto AI video"。
- **选人**: ReActor 本体偏最大脸, 多人画面需按输入脸列表或前置裁切; 复杂多人场景不如 FaceFusion reference 模式顺手。
- **成本**: 显存 ~2–4GB, 120 帧约几十分钟 (取决于修复模型)。
- **风险**: 同 128×128 上限; 在 ComfyUI 内跑会与 H3 抢显存, 建议换脸作为独立 pass 在 H3 释放显存后跑。
- 来源: [ComfyUI-ReActor GitHub](https://github.com/Gourieff/ComfyUI-ReActor) | [r/comfyui 视频换脸工作流帖](https://www.reddit.com/r/comfyui/comments/1rs4b13/i_built_a_face_swap_video_workflow_in_comfyui/) | ["Reactor Like A Boss" 参数帖](https://www.reddit.com/r/comfyui/comments/1gobugj/reactor_like_a_boss/)

### 1.3 roop-unleashed / ffmpeg remux 兜底模式
- **原理**: roop 的增强 fork, 批处理/多脸/保留源音轨; 但 **GitHub 仓库已被官方 disable (ToS 违规)**, 现在只能找镜像, 供应链风险高。
- **通用兜底**: 任何"输出无声"的换脸工具, 都用 ffmpeg stream copy 回填音轨, 不重编码:
  `ffmpeg -i no_audio_swap.mp4 -i original.mp4 -map 0:v:0 -map 1:a:0 -c copy -shortest final.mp4`
- **风险**: roop 系已死, 不建议新项目采用; FaceFusion 是其活跃继任者。
- 来源: [roop-unleashed 讨论帖 (仓库已禁)](https://www.reddit.com/r/StableDiffusion/comments/14n839u/roopunleashed/) | [ai-engine 帧抽取+remux 教程](https://ai-engine.net/blog/face-swap-video-python-tutorial)

**方向 1 小结**: 后处理换脸是**唯一不依赖任何生成模型自觉性**的方案 — 不管 i2v 把脸渲成什么样, 最后一 pass 统一钉脸。代价是 128×128 分辨率瓶颈 + 修复模型带来的"换脸感"。

---

## 方向 2: 关键帧端锁身份 (PuLID / InstantID 替换/叠加 IPAdapter-FaceID)

### 2.1 InstantID (SDXL, 现阶段 SDXL 端身份保真公认最强)
- **原理**: InsightFace (AntelopeV2) 人脸嵌入 + 人脸关键点 ControlNet 双通道注入 SDXL, 不需要训练。
- **效果 (量化)**: EditID 论文 (arXiv 2503.12526) 在中文人像基准上测得 **InstantID-SDXL Facesim (ArcFace 余弦) = 0.642–0.646**, 显著高于 PuLID-SDXL (0.384–0.399); 社区共识也是 InstantID 对真人照片相似度最好、可叠多张参考照。参照我们 0.28 的现状, 关键帧端换 InstantID 有望把"起点"提到 0.6+。
- **ComfyUI**: cubiq/ComfyUI_InstantID 成熟, 有 advanced 节点可分别调 InstantID 权重与 ControlNet 权重 (常用起点 0.8), 支持 Multi-ID、噪声注入、IPAdapter 联动。
- **成本/显存**: SDXL + antelopev2 + InstantID ControlNet 峰值 ~10–14GB, 12GB 卡贴线可跑 (开 --lowvram 或用 ComfyUI 新版动态 VRAM 管理, 分辨率压到 1024 内更稳)。
- **风险**: 与现有 IPAdapter face/body/scene 三 adapter 叠加时相互抢权重, 需重调调度 (我们已有 FACE adapter late-weakend 调度经验, commit 36c7532); InstantID 太高会钉死构图导致"证件照化", 与分镜多样性冲突。
- 来源: [ComfyUI_InstantID GitHub](https://github.com/cubiq/ComfyUI_InstantID) | [EditID 论文 (基准表)](https://arxiv.org/html/2503.12526v1) | [Stable Diffusion Art InstantID 指南](https://stable-diffusion-art.com/instantid/) | [IPAdapter_plus Issue #195 社区对比](https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/195)

### 2.2 PuLID-SDXL
- **原理**: 对比对齐损失 + 精确 ID 损失, 双分支 (Lightning T2I + 标准扩散) 训出的 tuning-free ID 模块, 卖点是"保身份的同时不污染底模型风格/可编辑性"。
- **效果 (量化)**: 同一 EditID 基准上 **PuLID-SDXL Facesim 仅 0.384–0.399**, 远低于 InstantID-SDXL; **PuLID-Flux 才到 0.724–0.735**。即: PuLID 的优势在 Flux 端, SDXL 端的身份保真反而不如 InstantID。
- **ComfyUI**: cubiq/PuLID_ComfyUI 支持 SDXL 与 Flux; KJNodes/balazik 的 PuLID 节点主要面向 Flux。
- **成本/显存**: PuLID-SDXL 约 8–10GB 可跑 (buffalo_l ~300MB + PuLID 模型 1–1.5GB 开销); PuLID-Flux 需 12–16GB+ 或量化。
- **风险**: 若为 PuLID 换 Flux 底模, 整个 kf 管线 (SDXL + IPAdapter 生态) 要推翻重来, 12GB 跑 FluxPuLID 也吃力; 不建议为此迁移。
- 来源: [PuLID 论文 (NeurIPS 2024)](https://arxiv.org/abs/2404.16022) | [PuLID_ComfyUI](https://github.com/cubiq/PuLID_ComfyUI) | [EditID 基准](https://arxiv.org/html/2503.12526v1)

### 2.3 IPAdapter-FaceID PlusV2 强化 (渐进改良, 不换架构)
- **原理**: 现有方案的升级版 — FaceID Plus V2 用 InsightFace 人脸嵌入 (而非 CLIP 图像嵌入), 对身份一致性显著更强; 官方模型卡也承认初代 FaceID 在 SDXL 上"实验性"。
- **效果**: 社区基准 (OSPA 论文) 测 IP-Adapter-FaceID 系相似度 **0.6–0.8**; SDXL FaceID + PlusFace 组合被 cubiq 仓库 issue 内评为"very good"; 多张参考图 (FaceID-Portrait 支持最多 5 张) 进一步稳定相似度。社区权重经验: LoRA 0.5–0.7, PlusV2 strength 0.7–1.0。
- **成本/显存**: 与现有 IPAdapter 相同, 增量 ~1GB, 12GB 无压力。
- **风险**: 提升是"量变"级别, 面对 H3 重渲可能仍不够 — 关键帧再像, i2v 端照样漂。
- 来源: [IP-Adapter-FaceID 官方模型卡](https://huggingface.co/h94/IP-Adapter-FaceID) | [SD Art IP-Adapter 指南](https://stable-diffusion-art.com/ip-adapter) | [Mikubill #2442 讨论](https://github.com/Mikubill/sd-webui-controlnet/discussions/2442)

**方向 2 小结**: 关键帧端从 IPAdapter-FaceID 换/叠 **InstantID 是 SDXL 生态内身份保真最优解 (0.64 vs 我们现状 0.28)**; PuLID 只在 Flux 端值得; 但要清醒: 关键帧像 ≠ 成片像, H3 重渲才是主要漂移源。

---

## 方向 3: 视频生成模型端 (保脸的 i2v / s2v)

### 3.1 H3 本地 R2V (ref2va) — 同一模型就有保参考能力
- **原理**: MiniMax 官方发布两套权重: FL2VA (T2V/I2V) 与 **Ref2VA (R2V, 参考图锁身份/风格/运镜/声音)**, 支持最多 9 张参考图 + 3 视频 + 3 音频。R2V 让模型在生成时直接吃角色参考, 而非只吃首帧。
- **效果**: 官方与社区定位 R2V 为"锁角色"工作流 (云端 H3 reference-to-video 被 Facebook ComfyUI 组评价为参考一致性出色); 云端 S2V-01 (前辈) 官方宣传"每帧忠于原图"。
- **成本/显存**: Ref2VA int8 checkpoint **20.97GB**, 12GB 卡必须重 offload; 参考案例: RTX 3060 12GB + 32GB RAM 跑 864×480 / 124 帧 / 20 步约 9 分钟 (重度 offload, 需大内存)。本地 GGUF 生态 (joeygambino/MiniMax-H3-GGUF) 已就绪。
- **风险**: 12GB 跑 R2V 会很慢且对系统 RAM 要求高 (32GB+); R2V 生成质量/运镜可控性 vs I2V 孰优需实测; H3 本地量化版重渲脸的量化损失本身没有公开量化对比。
- 来源: [ComfyUI 官方 H3 教程 (模型清单)](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) | [kingy.ai H3 本地指南 (12GB 实测)](https://kingy.ai/ai/ai-guides/minimax-h3-comfyui-local-guide/) | [MiniMax-H3-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-GGUF) | [MiniMax 官方 video-generation 文档](https://platform.minimax.io/docs/guides/video-generation)

### 3.2 H3 云端 API (reference-to-video, H3-Context-IR)
- **原理**: 官方 API 的参考生视频任务, 多模态参考 (最多 9 图 + 3 视频 + 3 音频) 注入, 全精度模型无量化损失。
- **效果**: fal/Runware/Evolink 等均上架; 云端全精度 + 官方参考机制, 重渲脸问题理论上比本地量化版轻 (无公开相似度数字, 但 r/comfyui 实测帖指出**即便是云端/本地 H3, 特写镜头身份 ~3 秒即漂, 过肩镜头能撑满 6.5 秒** — 景别比时长更影响漂移)。
- **成本**: 按量付费 (fal.ai minimax/h3/reference-to-video; 5–15s/条, 2K)。
- **风险**: 付费; 多角色参考会"串脸" (r/comfyui: 两个 ref 直接接上正常描述场景会 identity bleed, 需专门工作流); 台词/口型驱动与本地 h3_local voice 链路的衔接要重做。
- 来源: [fal.ai MiniMax H3 R2V](https://fal.ai/models/minimax/h3/reference-to-video) | [Runware H3 参考一致性指南](https://runware.ai/docs/models/minimax-h3/guides/reference-driven-consistency) | [r/comfyui H3 多角色一致性讨论](https://www.reddit.com/r/comfyui/comments/1vh0wcf/minimax_h3_multiple_character_consistency_which/) | [H3 身份漂移实测帖](https://www.reddit.com/r/comfyui/comments/1vm8j2c/measured_how_fast_identity_drifts_in_minimax_h3/)

### 3.3 开源保身份 i2v 研究模型 (ConsisID / Phantom / Stand-In / IPRO)
- **ConsisID** (PKU, CVPR 2025 Highlight): 频率分解 (低频全局身份 + 高频细节) 的 DiT 身份保持 t2v; 研究代码, 未产品化, 显存需求高, 无 ComfyUI 生态。
  来源: [项目页](https://pku-yuangroup.github.io/ConsisID/) | [GitHub](https://github.com/PKU-YuangGroup/ConsisID)
- **Phantom** (阿里通义, ICCV 2025): 跨模态对齐的 subject-consistent 视频生成, 基于 Wan 生态, 官方宣称严格保参考脸身份; 有 1.3B 轻量版可在 ~12GB 跑 480p, ComfyUI 工作流现成 (RunComfy/InstaSD)。**12GB 本地可行的另一条 i2v 替换路线**。
  来源: [项目页](https://phantom-video.github.io/Phantom/) | [GitHub](https://github.com/Phantom-video/Phantom) | [RunComfy 工作流](https://www.runcomfy.com/comfyui-workflows/comfyui-phantom-subject-to-video)
- **Stand-In** (微信 CVPR 2026): 只训 1% 额外参数的即插即用身份保持框架, 明确兼容 Wan + LoRA 社区生态; 太新, ComfyUI 集成少。
  来源: [GitHub](https://github.com/WeChatCV/Stand-In) | [论文](https://huggingface.co/papers/2508.07901)
- **IPRO** (arXiv 2510.14255): RL 奖励引导的身份保持 i2v, 纯研究。
  来源: [arXiv](https://arxiv.org/abs/2510.14255)
- 注: 任务里提到的 "DynVIL" 检索不到对应模型/论文, 疑似记忆偏差 (最接近的是 Dynamic-I2V, 但那是动态强度 i2v 不是保身份)。
- **共性风险**: 全部需要换掉 H3 → 运镜/音频/提示词链路重适配; 漫剧风格 (2.5D 漫画脸) 上这些真人向保身份模型的实际表现未知, 动漫画脸 InsightFace 嵌入质量也差。

**方向 3 小结**: 短期最现实的是**继续 H3 但改用 R2V 工作流** (本地 12GB 勉强 / 云端付费), 或者至少接受"特写镜 3 秒内切走"的实测规律; 中期 Phantom-1.3B 是 12GB 本地可试的保身份 i2v 平替。

---

## 方向 4: 社区实战 (r/comfyui, r/StableDiffusion, r/aifilmmaking)

- **共识 1 — 后处理换脸是短视频剧的标配兜底**: AI 短片圈普遍采用 "生成 → ReActor/FaceFusion 统一钉脸" 两段式; r/comfyui 有现成 "ReActor + CodeFormer 对 AI 生成视频换脸" 工作流 ([帖子](https://www.reddit.com/r/comfyui/comments/1rs4b13/i_built_a_face_swap_video_workflow_in_comfyui/))。
- **共识 2 — 景别管理 > 模型挣扎**: H3 身份漂移实测 ([帖子](https://www.reddit.com/r/comfyui/comments/1vm8j2c/measured_how_fast_identity_drifts_in_minimax_h3/)): 特写 ~3s 崩, 过肩镜可撑 6.5s。**分镜设计上避免长特写, 是零成本手段**。
- **共识 3 — 关键帧用"编辑型"模型保证跨镜同脸**: 社区流行链路是 Flux Kontext (或 ACE++/Vidu Q2) 把同一角色"编辑"进每个新分镜关键帧 (身份不漂), 再交给 Wan/H3 i2v ([Wan 2.2 FLF + Kontext 教程](https://www.youtube.com/watch?v=zYaHmscbWkY), [ACE++ & Wan 工作流](https://www.thinkdiffusion.com/consistent-character-aceplus-wan))。这正是我们 SDXL+IPAdapter kf 管线的同类思路, 可借鉴其"每镜关键帧都从同一锚定图 edit 而非重新生成"。
- **共识 4 — master shot 法**: 先生成一个"定妆全景", 再以它为参考派生各机位 ([r/generativeAI 多镜叙事讨论](https://www.reddit.com/r/generativeAI/comments/1pmqhso/exploring_multishot_storytelling_with_ai_how_do/)); 与我们"每集先定版角色立绘"一致, 但社区更进一步: **派生机位也走 img2img/edit 而非 txt2img**。
- **共识 5 — Wan 生态的 VACE/Stand-in**: r/comfyui 上 Wan 2.2 I2V 脸漂是高频问题 ([求助帖](https://www.reddit.com/r/comfyui/comments/1q9ku7o/wan_22_i2v_face_consistency_help/)), 社区答案是 VACE 参考层 / Stand-in 模型 / LTX Identity LoRA 等模型侧方案, 或干脆换 Vidu Q2 这类带 identity consistency 的商用模型 ([ComfyUI 博客 Vidu Q2](https://blog.comfy.org/p/vidu-q2-now-live-in-comfyui))。
- **共识 6 — 5 分钟以上一致性靠流程不靠模型**: r/Seedance_AI 讨论 5 分钟 AI 片难点是"一部电影感"而非单镜质量 ([帖子](https://www.reddit.com/r/Seedance_AI/comments/1um74jo/the_hard_part_of_a_5minute_ai_film_is_not_the/)) — 提示跨镜一致性要靠**锚定资产 + 统一后处理**的流程纪律解决。

---

## 推荐排序 (12GB 本地 + 已有 SDXL kf 管线)

> **修正 (基于附 0 仓库盘点)**: 原排序 1 的"装 FaceFusion"改为**修复自家 facelock_h3_ep.py 三坑** — 工具链全在位 (inswapper/insightface/后处理链), 修 remux 保音频 + 按 ref 选目标脸 + H3 调参约半天, 且不吃新依赖。FaceFusion 保留为备选 (若自修路线再翻车, 其 reference 模式 + 默认保音轨是现成替代)。

1. **【立即做, 零风险兜底】修自家的 `facelock_h3_ep.py` 三坑 (替代装 FaceFusion)**: ①末尾 ffmpeg remux 回填 H3 原音轨 (一行) ②目标脸从"最大脸"改"与 face_ref 嵌入最近脸" (~20 行) ③skin_and_lip 参数 CLI 化 + 对 H3 产物重调。上轮指标已证 0.28→0.88 可达, 修的是体验层三坑。注意 128×128 上限 — 特写大脸仍会糊, 需控制 restore 强度保画风。
2. **【同期做, 低成本提质】kf 端 IPAdapter-FaceID → InstantID (或 FaceID PlusV2 过渡)**: InstantID-SDXL 实测 Facesim 0.64+, 显存 10–14GB 贴线可跑; 关键帧起点越高, i2v 漂移后"落点"越好, 也给换脸 pass 更好的贴合底子。
3. **【分镜纪律, 零成本】景别管理**: H3 特写 3 秒即崩是实测规律 — 特写镜 ≤3s 或用过肩/中景; 锚定"定妆 master shot", 派生机位走 edit/img2img 而非重新 txt2img。
4. **【中期实验】H3 本地 R2V (ref2va)**: 官方就有保参考能力, 12GB+32GB RAM 可跑 (864×480, 9min/124帧); 用一两集 A/B 验证 R2V vs I2V 的脸漂差距。
5. **【按需付费】H3 云端 reference-to-video**: 全精度无量化损失 + 9 图参考; 若本地量化版被证实是漂移主因, 精修集切云端。
6. **【不推荐现在做】PuLID-SDXL / 换 Flux / ConsisID / IPRO**: PuLID-SDXL 身份保真 (0.38–0.40) 反不如 InstantID, 其优势在 Flux 端但 12GB 吃紧且要推翻 SDXL 生态; ConsisID/IPRO 无产品化。Phantom-1.3B 与 Stand-In 值得观望 (12GB 可跑 480p), 作为 H3 的中期备选。

**一句话路线**: 先上 FaceFusion 后处理兜底 (相似度问题当场止血), kf 端换 InstantID 抬起点, 分镜控特写时长; 然后用本地 R2V 做小规模 A/B, 数据说话再决定是否为精修集上云端。

---

## 附: 本报告引用源汇总

方向 1: [FaceFusion face-selector 文档](https://docs.facefusion.io/usage/cli-arguments/face-selector) | [FaceFusion output 文档](https://docs.facefusion.io/usage/cli-arguments/output-creation) | [LoopIn FaceFusion 指南](https://docs.loopin.network/tutorials/faceswap/facefusion-usage) | [ComfyUI-ReActor](https://github.com/Gourieff/ComfyUI-ReActor) | [r/comfyui ReActor 视频工作流](https://www.reddit.com/r/comfyui/comments/1rs4b13/i_built_a_face_swap_video_workflow_in_comfyui/) | [r/comfyui Reactor Like A Boss](https://www.reddit.com/r/comfyui/comments/1gobugj/reactor_like_a_boss/) | [r/FaceFusion inswapper 现状](https://www.reddit.com/r/FaceFusion/comments/1idoesk/is_inswapper_128_still_the_best_model_for_video/) | [roop-unleashed 讨论帖](https://www.reddit.com/r/StableDiffusion/comments/14n839u/roopunleashed/) | [ai-engine remux 教程](https://ai-engine.net/blog/face-swap-video-python-tutorial) | [FaceSwap Lab FAQ (128 限制)](https://glucauze.github.io/sd-webui-faceswaplab/faq/)

方向 2: [ComfyUI_InstantID](https://github.com/cubiq/ComfyUI_InstantID) | [PuLID_ComfyUI](https://github.com/cubiq/PuLID_ComfyUI) | [PuLID 论文](https://arxiv.org/abs/2404.16022) | [EditID 基准表](https://arxiv.org/html/2503.12526v1) | [IP-Adapter-FaceID 模型卡](https://huggingface.co/h94/IP-Adapter-FaceID) | [SD Art InstantID](https://stable-diffusion-art.com/instantid/) | [SD Art IP-Adapter](https://stable-diffusion-art.com/ip-adapter) | [IPAdapter_plus #195](https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/195) | [Mikubill #2442](https://github.com/Mikubill/sd-webui-controlnet/discussions/2442)

方向 3: [ComfyUI H3 官方教程](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) | [kingy.ai H3 本地指南](https://kingy.ai/ai/ai-guides/minimax-h3-comfyui-local-guide/) | [MiniMax-H3-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-GGUF) | [MiniMax 平台文档](https://platform.minimax.io/docs/guides/video-generation) | [fal.ai H3 R2V](https://fal.ai/models/minimax/h3/reference-to-video) | [Runware H3 指南](https://runware.ai/docs/models/minimax-h3/guides/reference-driven-consistency) | [r/comfyui H3 漂移实测](https://www.reddit.com/r/comfyui/comments/1vm8j2c/measured_how_fast_identity_drifts_in_minimax_h3/) | [r/comfyui H3 多角色](https://www.reddit.com/r/comfyui/comments/1vh0wcf/minimax_h3_multiple_character_consistency_which/) | [ConsisID](https://pku-yuangroup.github.io/ConsisID/) | [Phantom](https://phantom-video.github.io/Phantom/) | [Stand-In](https://github.com/WeChatCV/Stand-In) | [IPRO](https://arxiv.org/abs/2510.14255)

方向 4: [r/comfyui Wan2.2 脸一致求助](https://www.reddit.com/r/comfyui/comments/1q9ku7o/wan_22_i2v_face_consistency_help/) | [r/generativeAI 多镜叙事](https://www.reddit.com/r/generativeAI/comments/1pmqhso/exploring_multishot_storytelling_with_ai_how_do/) | [r/Seedance_AI 5分钟片一致性](https://www.reddit.com/r/Seedance_AI/comments/1um74jo/the_hard_part_of_a_5minute_ai_film_is_not_the/) | [ThinkDiffusion ACE++ & Wan](https://www.thinkdiffusion.com/consistent-character-aceplus-wan) | [Wan FLF + Kontext 教程](https://www.youtube.com/watch?v=zYaHmscbWkY) | [ComfyUI 博客 Vidu Q2](https://blog.comfy.org/p/vidu-q2-now-live-in-comfyui) | [SD Art 角色视频 face fix](https://stable-diffusion-art.com/fast-consistent-character-video2video/)
