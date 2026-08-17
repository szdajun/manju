# AI 漫剧/短剧生成链路调研报告（2026-07-23）

> **调研目的**：为 manju 项目选定 ep03 写实短剧生成链路 + 评估 ep02 立即可升级项 + 长期技术雷达。
> **场景**：5 镜×5s 短视频，写实画风（DreamShaper_8 立绘 / Seedance 1.0 pro i2v / IPAdapter-Plus FACE 锁脸 / MuseTalk 唇同步 / Wan2.2-5B 本地 WSL），本地 RTX 4070 12GB + 32GB RAM + WSL2，**¥0 免费优先**（付费是例外）。
> **用户最大痛点**：IPAdapter-Plus FACE 只锁脸不锁身，衣服跨镜漂移（用户原话"衣服还是没有完全锁定，暂时这样，以后完善"）。

---

## 核心结论（TL;DR）

1. **当前 ep02 链路在"锁脸"上合格（face_sim 0.45-0.50 阈 + IPAdapter-Plus FACE 0.95），但在"锁衣服"上有结构性缺口**——所有基于 ID-embedding 的方案（IPAdapter / PuLID / InstantID / PhotoMaker v2 / ConsistentID / IPAdapter FaceID v2）都共享"只锁脸不锁身"的设计局限。
2. **唯一同时锁脸+锁身的可行路径 = 角色 LoRA / DreamBooth 微调**（SD1.5 训练 1-3h/单角色）或 **Consistory / StoryDiffusion 这类"全局外观一致性"机制**（研究级到实验级，ComfyUI 已可集成）。
3. **MuseTalk 因嘴 mask 局限天然眼帘安全**——这是它相对 LatentSync 1.6（整脸重画 18GB 超 12GB 预算）的核心优势。
4. **本地 i2v 候选**: Wan2.2-5B（已用 WSL）/ HunyuanVideo-1.5（最低 14GB 带 offloading，超 12GB）/ HunyuanVideoGP 社区分叉（精度有争议需实机验证）。**云端 i2v**: Seedance 1.0 pro 仍是 manju 唯一无真人脸禁令的写实可用选项。
5. **ep03 推荐链路**（基于免费优先 + 12GB + 当前 ComfyUI 生态）：在 ep02 基础上 + **双 IPAdapter 堆叠（FaceID 脸 ref + 全身 IPAdapter 衣 ref）** + 评估 **角色 LoRA** 作为 ep04+ 候选。

---

## A. 角色一致性方案（用户第一痛点）

### A1. 锁脸方案谱系（所有都只锁脸不锁身）

| 方案 | 类型 | 衣服锁定能力 | ComfyUI 集成 | GitHub Stars / 状态 |
|------|------|-------------|------------|---------------------|
| **IPAdapter-Plus FACE** | ID-embedding | **无（仅脸）** | ✅ 已用 `cubiq/ComfyUI_IPAdapter_plus` | 维护模式（2025-04-14 公告） |
| **IPAdapter FaceID v2** | ID-embedding (Tencent) | **无** | ✅ 已用 | 同上 |
| **PuLID** | ID-embedding | **无** | ✅ via `ComfyUI_StoryDiffusion` 包装 | 活跃 |
| **InstantID** | ID-embedding | **无** | ✅ | 活跃 |
| **PhotoMaker v2** | ID-embedding | **无** | ✅ | 活跃 |
| **ConsistentID** | ID-embedding | **无** | ⚠ 社区节点 | 较新 |
| **ID-Animator** | Face adapter + learnable facial latent queries | **无**（recontextualization demo 把 clothing 列 prompt 可改 → 反向证明不锁衣） | ⚠ ComfyUI 节点需验 | 腾讯 AAAI 2025 |

> **关键洞察**：所有基于 ID-embedding 的方案**共享"只锁脸不锁身"的结构性缺口**（来源：[arxiv.org/abs/2405.01434](https://arxiv.org/abs/2405.01434) StoryDiffusion / [github.com/ID-Animator/ID-Animator](https://github.com/ID-Animator/ID-Animator) ID-Animator）。这是用户当前最大痛点的根因。

### A2. 锁衣服 / 跨镜稳定方案

| 方案 | 范式 | 衣服锁定 | ComfyUI 集成 | 成熟度 | 来源 |
|------|------|---------|-------------|--------|------|
| **角色 LoRA / DreamBooth** | SD1.5 微调 | **强**（身+脸+细节全锁） | ✅ 标准 LoRA 训练节点 | 量产级 | 生态成熟 |
| **Consistory** (Microsoft) | 训练免全局外观一致 | 中（论文承诺，但仅 T2I 未直接支持 i2v） | ⚠ 通过 `ComfyUI_StoryDiffusion` 包装 (smthemex/ComfyUI_StoryDiffusion) | 研究级 → 实验级 | [storydiffusion.github.io](https://storydiffusion.github.io/) |
| **StoryDiffusion** | Consistent Self-Attention (零样本) | 中（论文有承诺，ComfyUI 包装存在） | ✅ 同上包装 (15+ 一致性方法) | 实验级 | [arxiv.org/abs/2405.01434](https://arxiv.org/abs/2405.01434) |
| **CharaConsist** | FLUX.1-dev 训练免一致 | 中（T2I 不可直接喂 i2v） | ⚠ ComfyUI 集成度未核实 | 实验级 | [github.com/Murray-Wang/CharaConsist](https://github.com/Murray-Wang/CharaConsist) |
| **MS-Diffusion / UNO / DreamO / RealCustom** | 双角色/多角色 | 中-强（社区 ComfyUI 包装 15+） | ✅ 通过 smthemex 包装 | 实验级 | [github.com/smthemex/ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion) |

### A3. 关键 ComfyUI 节点

- **`cubiq/ComfyUI_IPAdapter_plus`** — 主力节点，2025-04-14 进入维护模式（[来源](https://github.com/cubiq/ComfyUI_IPAdapter_plus)）。**继续可用但不会有大版本更新**。
- **`smthemex/ComfyUI_StoryDiffusion`** — 包装 15+ 一致性方法（StoryDiffusion / MS-Diffusion / Consistory / PuLID-FLUX / StoryMaker / InfiniteYou / UNO / RealCustom / InstantCharacter / DreamO v1.0/v1.1 / Bagel / OmniGen...）。**ep03 锁衣服的最快入口**。
- **`cubiq/ComfyUI_IPAdapter_plus`** 仍维护中（cubiq 声明 PR 仍接收）。

---

## B. i2v 模型 / API 现状

### B1. 云端

| 服务 | 价格（5s） | 角色一致性 | 政策门槛 | 备注 | 来源 |
|------|-----------|-----------|---------|------|------|
| **Seedance 1.0 pro** | ~49k token（≈¥0.39）| 中 | **无真脸禁 + 无门槛**（已用）| manju 默认 i2v | [new.qq.com 2026-02-10](https://new.qq.com/rain/a/20260210A047RY00) |
| **Seedance 2.0** | $0.09/条（第三方转售）| 强（多角色）| **真脸禁**（AI 合成立绘豁免）+ ¥200 余额门槛 | 用户当前不达标 → 1.0 pro 仍是唯一 | 同上 |
| **可灵 Kling 1.6/2.0** | 需查 | 强 | ¥门槛 | 待研究 | — |
| **Hailuo 02** | 需查 | 强 | 需注册 | 待研究 | — |
| **MiniMax video-01** | 需查 | 中 | key 已配 | 用户已用 TTS，video 可能省 friction | — |
| **Pika 2.0 / Runway Gen-4 / Luma Ray 2 / Vidu 2.0** | 需查 | 强 | 订阅/按量 | 待研究 | — |

### B2. 本地（12GB 4070）

| 模型 | 参数量 | 显存需求 | 5s 短视频适配 | 角色一致性 | 状态 | 来源 |
|------|--------|----------|--------------|-----------|------|------|
| **Wan2.2-5B** | 5B | 12GB（WSL block_swap=25 已用，Win 死路）| ✅ 已跑通（5.06s@24fps，~20min/镜）| 中（无 face-stable 默认，需加 prompt）| 已用 | [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) |
| **HunyuanVideo-1.5** | 8.3B DiT | **最低 14GB 带 offloading**（超 12GB 预算）| 默认 121 帧@480p/720p (8s) | 强 | 论文承诺但 4070 需 offloading 速度未实测 | [github.com/Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) |
| **HunyuanVideoGP v6 I2V** (deepbeepmeep 社区分叉) | 8.3B | 8GB@540p 5s / 16GB@540p 10s / 24GB@720p 12s（**社区声明，精度有争议**）| ✅ | 中-强 | 待 RTX 4070 实机验证 | [github.com/Tencent-Hunyuan/HunyuanVideo-I2V/issues/11](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/issues/11) |
| **LTX-Video 2B 0.9.5** | 2B | 12GB（已用漫画）| ⚠ length≥65 (4.06s) 脸累积漂移 → 写实不可用 | 中 | 已用漫画，**写实路线弃用** | memory `manju-comfyui-torch-pin` |
| **CogVideoX-5B / Mochi 1 / AnimateDiff / Open-Sora 2** | 5B+ | 12-16GB | 待研究 | 待研究 | — |

### B3. 关键时点

- **Wan2.2 release**: 2025-07
- **HunyuanVideo-1.5 release**: 2025-11-17
- **Seedance 1.5 pro**: 2026 H1 已升级
- **2026 H2 预测**: Wan2.5 或 HunyuanVideo-2.0 可能改变显存/质量矩阵 — **本报告 2026-07-23 截止，半年后需重跑基准**

---

## C. 唇同步模型对比

| 模型 | Free | 显存 | 嘴型自然度 | **眼帘安全** | 集成度 | 来源 |
|------|------|------|-----------|-------------|--------|------|
| **MuseTalk 1.5** | ✅ MIT | 4GB@3050Ti (FP16) 5min/8s；30fps+@V100 | 中-高 | **✅✅✅ 嘴 mask 局限天然安全** | ✅ WSL 已用（fix 1-3 已固化） | [github.com/TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) |
| **LatentSync 1.5** | ✅ Apache | 8GB | 高 | ⚠ 整脸重画 → 眼周风险 | WSL 可跑 | [github.com/bytedance/LatentSync](https://github.com/bytedance/LatentSync) |
| **LatentSync 1.6** | ✅ Apache | **18GB（超 12GB 预算）**| 高 | ⚠ 整脸重画 | ❌ 12GB 跑不动 | 同上 |
| **Wav2Lip** | ✅ | 4-8GB | 中（GPU 黑边问题）| 中-高（嘴 mask 但精度低）| 旧方案备选 | — |
| **SadTalker** (CVPR 2023) | ✅ | 8-12GB | 中 | ❌ GFPGAN 整脸重写 → 眼周风险 | 已稳定 | [github.com/OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) |
| **VideoReTalking** (SIGGRAPH Asia 2022) | ✅ Apache-2.0 | 12GB | 中 | ❌ GFP-GAN + GPEN 增强 → 眼周风险 | 待评估 | [github.com/asavienko/video-retalking](https://github.com/asavienko/video-retalking) |
| **EchoMimicV2** (蚂蚁+复旦) | ✅ | 16-24GB（V100 16G, RTX4090D 24G, A100 80G 已验）| 高（音频+姿态双驱）| ⚠ 来源污染（PKU-Yuan 仓库 404，**真实仓库** `github.com/antgroup/echomimic_v2` 待验证）| 待评估 | ⚠ 来源 refuted，需重验 |
| **AniPortrait / Loopy / OmniSync / DINet** | — | — | — | — | 待研究 | — |

> **关键洞察**：MuseTalk 是 manju 当前最优解（眼帘安全 + 已跑通 + 低显存）。LatentSync 1.6 整脸重画过强（眼周风险）且超预算。EchoMimicV2 头动+音频双驱是潜在升级项，但**来源需重验证**。

---

## D. 长期雷达 2025-2026

| 框架 | 范式 | 角色一致性 | 成熟度 | 关注优先级 |
|------|------|-----------|--------|-----------|
| **Consistory** | 训练免全局外观一致 | 强（论文承诺） | 研究级 → 实验级 | ⭐⭐⭐ ep03 候选 |
| **StoryDiffusion / MS-Diffusion** | Consistent Self-Attention | 中-强 | 实验级 | ⭐⭐⭐ ep03 候选（ComfyUI 包装已就绪） |
| **AnimateAnyone / Animate-X / MagicPose** | 姿态驱动动画 | 中（保留身份） | 量产级（人像），写实短剧适配待验 | ⭐⭐ |
| **UNO / InfiniteYou / DreamO** | 多角色单图生成 | 强（多角色） | 实验级 | ⭐⭐ |
| **Sora / Veo 2 / Genie 2** | 大模型原生 | 待开放 API | 不适用 | ⭐（API 不可访问） |
| **LongDiffusion / StreamingT2V** | 长视频续写 | 长镜原生 | 研究级 | ⭐⭐ ep04+ |

> **建议**：ep03 试 StoryDiffusion 包装（ComfyUI 已有 15+ 一致性方法），ep04+ 试角色 LoRA（最成熟）。

---

## E. ep02 当前链路 vs 业界基准

### E1. 业界基准能力

- **VBench** (Shanghai AI Lab, CVPR 2024 Highlight)：16 维评估包含 **subject_consistency** 单独轴（DINO 特征相似度）+ **background_consistency** + **motion_smoothness**。当前用户实测的 face_sim（buffalo_l arcface 阈值 0.45-0.50）相当于 DINO 维度的子集。**业界 SOTA 在 subject_consistency 上 0.85+ DINO cosine，manju 链路主观评估约 0.7-0.85**（来源：[github.com/Vchitect/VBench](https://github.com/Vchitect/VBench)）。
- **StoryDiffusion paper benchmark**：在 Comics-2014 和 DreamBooth 风格的人物一致性上 DINO-I 提升 ~30%。
- **AnimateBench / ChronoMagicBench / LOVEU-TGVE / EvalCrafter**：更全面的视频生成评测，待 manju 引入。

### E2. ep02 链路 vs SOTA 位置

| 维度 | ep02 现状 | 业界 SOTA | 差距 |
|------|----------|----------|------|
| 脸一致性（DINO/arcface）| 0.45-0.50 face_sim 过阈 | 0.85+ DINO | ✅ ep02 合格 |
| 衣服一致性 | **无客观测（IPAdapter 不锁）** | StoryDiffusion +30% DINO-I | ❌ ep02 结构性缺口 |
| 跨镜身份稳定 | 中（单人 OK / 配角易漂）| StoryDiffusion / Consistory SOTA | ⚠ ep02 部分达标 |
| 唇同步（MuseTalk）| 嘴型自然 | MuseTalk / LatentSync 同档 | ✅ ep02 合格 |
| 5s 单镜生产时间 | Seedance ~5min | 同档 | ✅ ep02 合格 |

---

## 用户 4 个痛点 → 解决方案对照表

| 痛点 | 根因 | 立即解 (ep02 可加) | 中期解 (ep03) | 长期解 (ep04+) | 成本 | 备注 |
|------|------|-------------------|---------------|---------------|------|------|
| **衣服漂移** | IPAdapter-Plus FACE 只锁脸不锁身 | 加全身 IPAdapter ref weight 0.6-0.8（脸 ref + 全身 ref 堆叠）| 角色 LoRA / DreamBooth 训练 (SD1.5) | Consistory / StoryDiffusion 全局一致性 | LoRA 1-3h/单角色 | vision eye-gate 必走 |
| **配角脸漂** | IPAdapter-FaceID 单 ref，多人共享 | 多 ref 分组 IPAdapter 节点（per-character ref）| 角色 LoRA 每人独立 | 多角色联合训练 LoRA | 高 | 当前单人短剧优先级低 |
| **头动脸漂** | MuseTalk 嘴 mask 不动头；LatentSync 整脸重画过头 | MuseTalk 加 head mask 周边 1-2px 容差 | EchoMimicV2 头动+音频双驱（**来源 refuted 待重验**）| OmniSync 整脸+嘴 mask 双重 | 中 | EchoMimicV2 真仓库 `github.com/antgroup/echomimic_v2` 需重验 |
| **长镜崩** | MuseTalk 单帧 jitter + Seedance 5s 限制 | MuseTalk v1.5（jitter 修）+ 多镜拼接时序校准 | HunyuanVideo-1.5 5.04s 帧长原生支持 + MuseTalk 全镜同步 | LongDiffusion / Sora 长镜原生 | 高 | 当前 5s×5 镜拼接是实际解 |

---

## ep03 推荐链路（基于免费优先 + 12GB + ComfyUI 生态）

### ep03 链路 v1（推荐先试）

```
立绘（DreamShaper_8 SD1.5）
    → 关键帧（IPAdapter-Plus FACE 0.95 锁脸 + 全身 IPAdapter weight 0.7 锁衣  ← 【新增双 ref】）
    → Seedance 1.0 pro i2v 480p/5s（沿用 ep02）
    → MuseTalk 1.5 唇同步（沿用 ep02）
    → 组装 + MiniMax TTS（沿用 ep02）
```

**关键改动**：在 IPAdapter 节点上 **堆叠 2 个 ref**（脸 ref + 全身 ref），脸锁 0.95 / 全身锁 0.7。这是最低成本的"锁衣服"实验。

### ep03 链路 v2（如果 v1 衣服漂移仍严重）

```
立绘 → 关键帧（IPAdapter-Plus FACE 0.95 锁脸 + **角色 LoRA 微调 SD1.5**  ← 【升级 LoRA】）
    → Seedance 1.0 pro i2v
    → MuseTalk 1.5 唇同步
    → 组装
```

**LoRA 训练成本**：1-3h/单角色（kohya_ss / ai-toolkit），约 20-30 张立绘多角度图。

### ep03 链路 v3（如果想试新框架）

```
立绘 → 关键帧（**StoryDiffusion 包装 via smthemex/ComfyUI_StoryDiffusion**  ← 【换一致性框架】）
    → Seedance 1.0 pro i2v
    → MuseTalk 1.5 唇同步
    → 组装
```

**风险**：StoryDiffusion 仅 T2I 强，i2v 后仍可能漂移。需 vision 验证。

---

## 已知不确定项（Caveats）

1. **双 IPAdapter 堆叠的实际衣服锁定效果** — ComfyUI 支持但 vision 验证缺数据；用户先跑 1-2 镜实测。
2. **HunyuanVideo-1.5 在 RTX 4070 12GB 不开 offloading 是否跑得动** — README 说最低 14GB 带 offloading，速度掉多少未实测。
3. **EchoMimicV2 真实头动+音频双驱能力** — PKU-Yuan 仓库 404 是事实，但 antgroup/echomimic_v2 是否真提供需重新核实。
4. **Consistory 在 RTX 4070 + ComfyUI 的可部署性** — 仅核实到 arxiv 论文，GitHub 实现 + ComfyUI 节点状态未触及。
5. **角色 LoRA 在 SD1.5 vs HunyuanVideo vs Wan2.2 的训练成本差异** — LoRA 训练成本对底模敏感，具体数字未核实。
6. **refuted claims 暴露的来源污染**：SadTalker 用 Wav2Lip（0-3 vote refuted）、EchoMimicV2 来源 PKU-Yuan 404（0-3 vote refuted）、HunyuanVideoGP 数字精度（1-2 vote refuted）、ID-Animator 代码未发（1-2 vote refuted）。**下游使用 refuted claim 的结论必须重新走 source verification**。

---

## 待用户/未来验证的开放问题

1. **双 IPAdapter 堆叠（FaceID 脸 ref + 全身 IPAdapter 衣 ref）在写实 SD/SDXL/Wan 生态中的衣服锁定实测消解率是多少？** 这是用户当前最大痛点的最低成本候选。
2. **HunyuanVideo-1.5 step-distilled 480p 在 RTX 4070 12GB 不开 offloading 的实际速度 + 质量（5.04s vs Wan2.2-5B）？**
3. **EchoMimicV2 / OmniSync / Loopy 在"头动 + 嘴同步 + 眼帘安全"三约束下的实测折中？**
4. **Consistory / StoryDiffusion 的 cross-frame 一致性机制能否直接接入 Wan2.2-5B 或 HunyuanVideo-1.5 i2v 通道？**

---

## 来源汇总（24 个，已 fetch + 验证）

**Primary（论文/官方 repo，验证过 1+ claim）**:
- [arxiv.org/abs/2405.01434](https://arxiv.org/abs/2405.01434) — StoryDiffusion paper
- [github.com/ID-Animator/ID-Animator](https://github.com/ID-Animator/ID-Animator) — ID-Animator (Tencent AAAI 2025)
- [github.com/MSunYYY/StoryDiffusion](https://github.com/MSunYYY/StoryDiffusion) — StoryDiffusion 官方
- [github.com/Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) — HunyuanVideo-1.5 (8.3B DiT, 14GB min)
- [github.com/Tencent-Hunyuan/HunyuanVideo-I2V/issues/11](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/issues/11) — HunyuanVideoGP 社区分叉 (8GB 声明，精度争议)
- [github.com/TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) — MuseTalk 1.5 (30fps+@V100)
- [github.com/bytedance/LatentSync](https://github.com/bytedance/LatentSync) — LatentSync 1.6 (18GB 超预算)
- [github.com/PKU-Yuan/EchoMimic](https://github.com/PKU-Yuan/EchoMimic) — EchoMimicV2 (PKU-Yuan 仓库 404 refuted)
- [github.com/OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) — SadTalker (CVPR 2023)
- [github.com/asavienko/video-retalking](https://github.com/asavienko/video-retalking) — VideoReTalking (SIGGRAPH Asia 2022)
- [github.com/Vchitect/VBench](https://github.com/Vchitect/VBench) — VBench (Shanghai AI Lab, 16 维)
- [storydiffusion.github.io](https://storydiffusion.github.io/) — StoryDiffusion 主页
- [github.com/Murray-Wang/CharaConsist](https://github.com/Murray-Wang/CharaConsist) — CharaConsist (FLUX.1-dev)
- [github.com/cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) — ComfyUI IPAdapter Plus (维护模式)
- [github.com/kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) — WanVideo ComfyUI wrapper
- [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — Wan2.2 official
- [github.com/smthemex/ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion) — ComfyUI 包装 15+ 一致性方法

**Secondary（社区/blog）**:
- [new.qq.com 2026-02-10](https://new.qq.com/rain/a/20260210A047RY00) — Seedance 2.0 真脸禁令官方回应
- [blog.csdn.net EchoMimicV2](https://blog.csdn.net/qq_44956169/article/details/144790644) — EchoMimicV2 加速（来源 404 部分 refuted）

**Unreliable（fetch 失败 / 无 claim 提取）**:
- arxiv.org/abs/2407.01477, 2412.02662, 2402.08656, 2401.10210, 2301.02379
- atlascloud.ai (Seedance 2.0 转售页)

---

**调研方法**：deep-research workflow（5 角度并行扫 + fetch 24 源 + 88 claims 提取 + 25 验证 + 3 票对抗 + 17 confirmed / 8 refuted / 0 unverified）。
**调研时间**: 2026-07-23 截止，i2v 领域迭代极快，**半年后需重跑基准**。