"""ep03《装备先到位》分镜剧本 — 健身房日常幽默, 单人, 林夏主角。

主题: 林夏第一次认真去健身房跑步, 装备非常专业(包/毛巾/手表/手机), 热身调设备
  一气呵成, 最后却先打开手机拍照发朋友圈, 形成轻幽默反转 ("先拍照证明我来过, 跑步稍后再说")。
叙事弧: 钩子(走进健身房) → 起势(热身) → 转折(整理装备) → 反转(自拍) → 收尾(CTA 跑步约定)。
  5 镜完成一个完整回合, 单人视角。

技术约束(钉死):
- 写实风格 (REAL_STYLE), 沿用 ep02 写实路径 (DreamShaper_8 + IPAdapter + Seedance 1.0 pro + MuseTalk + MiniMax TTS)
- 单人, 不复用 ep02 小雅/婷婷 (验证通用性)
- 关键帧 480×832 (Wan i2v 原生桶, ep02 已验证)
- 5 镜 × 5s × 9:16 = ~25s 成片
- 主题: 验证"双 IPAdapter 堆叠 (脸 ref 0.95 + 全身 ref 0.70) 锁衣服"效果
- prompt 来源说明: 因用户 ARK key 无 Doubao 文本模型权限 (2026-07-23 实测),
  I2V_MOTION 全部手写, 沿用 ep02 I2V_MOTION_REAL_OVERRIDE 范式 + OpenMontage
  ai-video-gen 7 步最佳实践 (descriptive + camera/lighting/style/mood + aspect + ref img + 5min timeout)。
  后续如需 LLM 优化, 可补 OpenAI/Claude key 或自部署开源 LLM。

锁衣服设计核心: 服装指纹必须**每镜重复**, 让 IPAdapter 全身 ref weight 0.7 持续
  锁住 (上衣雾蓝色 + 左胸白色弧线标志 / 裤子黑色高腰 + 裤侧灰线 / 白色跑鞋)。
  这是用户最大痛点 "IPAdapter-Plus FACE 只锁脸不锁身" 的实证实验。

**豆包推理接入点 (2026-07-23 验证)**: `ep-20260723170404-g9xdl`, 底层 `doubao-seed-1.6-250615`
  支持 chain-of-thought (reasoning_tokens~1500), 用于 prompt 优化。
  ARK key 之前的 doubao-pro-* 文本模型 404 → 此接入点是唯一可用文本 LLM。
  模板: `scripts/_build_ep03_motion_prompt.py` (offline batch, 见 git history)。
"""

# 林夏角色指纹(每镜必带; 同时喂 IPAdapter 全身 ref 0.7 让衣服跨镜一致)
# 钉死的服装指纹 (Plan agent 拍板, 雾蓝上衣 / 黑色高腰 / 白跑鞋 / 高马尾)
CHAR = ("1girl, solo, Linxia (林夏), a slim-fit 26-30 year-old Chinese woman, "
        "black hair in a high ponytail, "
        "wearing a misty-blue long-sleeve athletic top with a small white arc logo on the left chest, "
        "black high-waist athletic pants with a thin gray side stripe down the leg, "
        "white running shoes, "
        "natural makeup, bright energetic eyes, friendly confident expression, "
        "modern athletic wear, gym setting")

# 写实风格 (沿用 ep02 REAL_STYLE; 反逼漫画风)
REAL_STYLE = ("photorealistic, real photograph, 50mm lens, cinematic lighting, shallow depth of field, "
              "detailed skin texture, natural colors, sharp focus, vertical composition, 8k uhd")

# 写实 NEG (沿用 ep02 REAL_NEG; 加反厚下眼睑 + 反换衣 / 反色变)
REAL_NEG = ("text, watermark, logo, signature, extra fingers, deformed hands, mutated hands, "
            "bad anatomy, extra limbs, blurry, lowres, jpeg artifacts, "
            "anime, manga, cartoon, illustration, painting, 3d render, cg, lineart, flat shading, "
            "plastic skin, airbrushed, multiple views, reference sheet, "
            "face morphing, face distortion, warped face, melting face, asymmetrical eyes, deformed face, "
            "puffy under-eye, heavy eye bags, thick lower eyelid, prominent eye bags, dark circles, tired eyes, "
            "color shift, clothing color change, outfit change, white shirt, blue shirt, different clothes")

# EYE_FIX (沿用 ep02; 防 i2v 眼睛退化)
EYE_FIX = "bright fresh energetic eyes, smooth under-eye area, no eye bags, natural well-formed eyelids"

# 关键帧静态场景 (5 镜, 每镜必带服装指纹词)
# OpenMontage 7 步最佳实践: 描述性 + camera + lighting + style + mood
SHOTS = {
    1: ("medium full shot, modern gym interior with treadmill in the background, "
        "Linxia wearing misty-blue long-sleeve athletic top with white arc logo on left chest, "
        "black high-waist athletic pants with gray side stripe, white running shoes, "
        "black hair in a high ponytail, "
        "stands at the entrance with a small gym bag, looks around confidently, "
        "bright overhead gym lighting, natural colors, vertical composition, modern fitness atmosphere"),
    2: ("medium shot, knee-up framing, "
        "Linxia in misty-blue long-sleeve athletic top with white arc logo, "
        "black high-waist athletic pants with gray side stripe, "
        "doing gentle shoulder rolls to warm up, focused determined expression, "
        "soft warm gym lighting, shallow depth of field, natural skin texture, modern gym interior"),
    3: ("three-quarter body shot, side angle, "
        "Linxia in misty-blue long-sleeve athletic top with white arc logo on left chest clearly visible, "
        "black high-waist athletic pants with gray side stripe down the leg clearly visible, "
        "white running shoes, "
        "carefully placing a white towel on the treadmill console and adjusting her sports watch, "
        "soft side lighting from gym window, calm preparatory mood, "
        "face clearly visible and expressive, modern gym setting"),
    4: ("medium shot, "
        "Linxia in misty-blue long-sleeve athletic top with white arc logo, "
        "black high-waist athletic pants with gray side stripe, "
        "steps onto the treadmill slowly with two careful steps, hand on the console rail, "
        "concentrated cautious expression, "
        "even gym lighting, modern fitness mood, face clearly visible front-facing"),
    5: ("medium close-up, waist-up framing, "
        "Linxia in misty-blue long-sleeve athletic top with white arc logo on left chest, "
        "black high-waist athletic pants (only top edge visible), "
        "holding up her smartphone at arm's length adjusting the angle for a selfie, "
        "satisfied playful smirk, "
        "warm flattering front lighting on her face, bokeh gym background, "
        "selfie-taking comedic mood, face clearly visible and smiling"),
}

# 台词 (中文字幕/旁白; 全部林夏一人, SPEAKER 单一)
LINES = {
    1: "今天开始, 我要做一个真正自律的人。",         # 钩子: 进入健身房, 决心
    2: "第一步, 先把状态调整到专业模式。",           # 起势: 热身
    3: "水, 毛巾, 音乐...... 一个都不能少。",       # 转折: 整理装备 (重点验证锁衣服镜)
    4: "好了, 现在只差最后一个步骤。",                # 反转铺垫
    5: "先拍张照, 证明我来过。",                      # 反转落点 (CTA 跑步约定延后)
}

# i2v 运动意图 (Seedance 1.0 pro; 每镜强调"服装指纹不变"+"低幅度"+"脸稳定"+"嘴可见")
# 范式参考: ep02 I2V_MOTION_REAL_OVERRIDE (写实路径已验证)
# + OpenMontage 7 步最佳实践: camera + lighting + style + mood
# 优化方式: 豆包推理模型 (ep-20260723170404-g9xdl, 底层 doubao-seed-1-6-250615) +
#   few-shot (ep02 I2V_MOTION_REAL_OVERRIDE) + shot-aligned prompt 派生 (见 _build_ep03_motion_prompt)
# 2026-07-23 跑通, reasoning_tokens~1500, output_tokens~1700
#
# 双源混合: shot1 取 v1 版 (LLM 误改 head turn), shot2-5 取 v2 版 (LLM 与 SHOTS 对齐更准)
I2V_MOTION = {
    1: ("Linxia walks slowly into the gym, adjusting her high ponytail's elastic with one hand, "
        "subtle step motion, "
        "the misty-blue top stays misty-blue throughout, "
        "the black pants stay black throughout, "
        "the white arc logo stays in the same position on the left chest"),
    2: ("She rolls her shoulders forward in a circular motion, then reverses to roll them backward, "
        "subtle warm-up motion, focused determined expression, "
        "the misty-blue top stays misty-blue throughout, "
        "the black pants stay black throughout, "
        "the white arc logo stays in the same position on the left chest"),
    3: ("She smooths the towel onto the console with her right hand, "
        "then adjusts her sports watch with her left, "
        "slow careful methodical movements, calm preparatory mood, "
        "the misty-blue top stays misty-blue throughout, "
        "the black pants stay black throughout, "
        "the white arc logo stays in the same position on the left chest"),
    4: ("She lifts her right foot onto the treadmill, then her left foot, "
        "and gently grips the handrails, "
        "very subtle low-amplitude motion, concentrated cautious expression, "
        "the misty-blue top stays misty-blue throughout, "
        "the black pants stay black throughout, "
        "the white arc logo stays in the same position on the left chest"),
    5: ("She tilts the smartphone slightly up and down, "
        "shifting her gaze between the screen and camera, "
        "subtle playful motion, satisfied smirk gradually forming, "
        "the misty-blue top stays misty-blue throughout, "
        "the black pants stay black throughout, "
        "the white arc logo stays in the same position on the left chest"),
}

# SPEAKER 路由 (单人 = 全 LINXIA; MiniMax voice ID 复用 ep02)
# voice_ep03.py 实际用 MINIMAX_VOICE env; 这里只标角色
SPEAKER = {
    1: "LINXIA",
    2: "LINXIA",
    3: "LINXIA",
    4: "LINXIA",
    5: "LINXIA",
}

# ---- 角色资产 (2026-08-06 orchestrator 接入: 从生成器提进分镜数据, 让 run_ep 能按剧本驱动) ----
TITLE = "装备先到位"
DURATION_S = 5          # 单镜时长 (Seedance i2v 默认)
FACE_REF = "output/ep03_lihui/linxia_real_face.png"   # 脸 ref (crop_face 产出)
BODY_REF = "output/ep03_lihui/linxia_real.png"        # 全身 ref (锁衣服)
VOICE_ID = "female-shaonv"                            # MiniMax 女声 (voice_ep03 用 MINIMAX_VOICE env, 默认一致)