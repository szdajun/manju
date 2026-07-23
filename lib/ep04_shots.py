"""ep04 双角色 LoRA 剧本 (2026-07-23)。
   小雅 5 镜写字楼剧本 + 林夏 5 镜健身房剧本, 验 LoRA 治本 (ep02 小雅裤装 / ep03 林夏裤色+白弧标志跨镜一致)。

主题对照:
- XIAOYA: 写字楼办公室 5 镜 (上班/开会/午餐/报告/下班)
  服装: 白衬衫 + 铅笔裙 + 长直黑发 (ep02 已用, 沿用)
- LINXIA: 健身房 5 镜 (入场/热身/深蹲/拉伸/收尾拍照)
  服装: 雾蓝长袖+白弧标志 + 黑高腰+灰侧线 + 白鞋 + 高马尾 (ep03 已用, 沿用)

服装指纹 (钉死):
- xiaoya: 白衬衫 (white office blouse), 铅笔裙 (pencil skirt), 长直黑发 (long straight black hair)
- linxia: 雾蓝长袖+白弧, 黑高腰+灰侧线, 白鞋, 高马尾
触发词: xiaoya (见 _aug_lihui_dataset.py) / linxia (见 _aug_linxia_dataset.py)
"""

# 通用风格 + 反厚下眼睑 + 反换衣
REAL_STYLE = (
    "photorealistic, real photograph, 50mm lens, cinematic lighting, "
    "shallow depth of field, detailed skin texture, natural colors, sharp focus, "
    "vertical composition, 8k uhd"
)

REAL_NEG = (
    "text, watermark, logo, signature, extra fingers, deformed hands, mutated hands, "
    "bad anatomy, extra limbs, blurry, lowres, jpeg artifacts, "
    "anime, manga, cartoon, illustration, painting, 3d render, cg, lineart, flat shading, "
    "plastic skin, airbrushed, multiple views, reference sheet, "
    "face morphing, face distortion, warped face, melting face, asymmetrical eyes, deformed face, "
    "puffy under-eye, heavy eye bags, thick lower eyelid, prominent eye bags, dark circles, tired eyes, "
    "color shift, clothing color change, outfit change, different clothes"
)

EYE_FIX = "bright fresh energetic eyes, smooth under-eye area, no eye bags, natural well-formed eyelids"

# Per-character CHAR (基础角色描述, 含服装 + 脸)
CHAR_XIAOYA = (
    "1girl, solo, slim fit young Chinese woman, long straight black hair, "
    "bright cheerful face, confident smile, white office blouse, pencil skirt, office attire"
)
CHAR_LINXIA = (
    "1girl, solo, slim fit young Chinese woman age 26-30, black hair in high ponytail, "
    "bright energetic eyes, "
    "misty-blue long-sleeve athletic top with white arc logo on left chest, "
    "black high-waist athletic pants with thin gray side stripe, "
    "white running shoes, fitness attire"
)

CHAR_BY_NAME = {"xiaoya": CHAR_XIAOYA, "linxia": CHAR_LINXIA}

# Back-compat: 默认 CHAR = xiaoya (用于 ep04_shots 旧 import path)
CHAR = CHAR_XIAOYA

REAL_STYLE = (
    "photorealistic, real photograph, 50mm lens, cinematic lighting, "
    "shallow depth of field, detailed skin texture, natural colors, sharp focus, "
    "vertical composition, 8k uhd"
)

REAL_NEG = (
    "text, watermark, logo, signature, extra fingers, deformed hands, mutated hands, "
    "bad anatomy, extra limbs, blurry, lowres, jpeg artifacts, "
    "anime, manga, cartoon, illustration, painting, 3d render, cg, lineart, flat shading, "
    "plastic skin, airbrushed, multiple views, reference sheet, "
    "face morphing, face distortion, warped face, melting face, asymmetrical eyes, deformed face, "
    "puffy under-eye, heavy eye bags, thick lower eyelid, prominent eye bags, dark circles, tired eyes, "
    "color shift, clothing color change, outfit change, different clothes"
)

EYE_FIX = "bright fresh energetic eyes, smooth under-eye area, no eye bags, natural well-formed eyelids"

# 小雅 5 镜 SHORT — 写字楼日常
SHOTS_XIAOYA = {
    1: ("medium shot, the slim fit woman standing at office reception desk, holding a folder, "
        "looking at the camera with a confident smile, modern open office, "
        "wearing white office blouse, pencil skirt, long straight black hair"),
    2: ("medium close-up, the slim fit woman sitting at conference table, gesturing with one hand, "
        "presenting to colleagues, confident expression, modern conference room, glass walls, "
        "wearing white office blouse, pencil skirt, long straight black hair"),
    3: ("medium shot, the slim fit woman eating lunch at office cafeteria, holding chopsticks, "
        "smiling at camera, bright cafeteria, modern office, "
        "wearing white office blouse, pencil skirt, long straight black hair"),
    4: ("medium close-up, the slim fit woman at desk presenting a report, holding papers, "
        "speaking confidently, modern office desk with monitor, "
        "wearing white office blouse, pencil skirt, long straight black hair"),
    5: ("medium shot, the slim fit woman at office door, picking up handbag, "
        "waving goodbye to colleagues, cheerful expression, "
        "wearing white office blouse, pencil skirt, long straight black hair"),
}

# 林夏 5 镜 SHORT — 健身房日常 (验证 LoRA 对裤色+白弧的治本能力, ep03 同病)
SHOTS_LINXIA = {
    1: ("medium shot, the athletic woman walking into a bright modern gym, "
        "looking at the camera with a bright confident smile, gym reception area, "
        "wearing misty-blue long-sleeve athletic top with small white arc logo on left chest, "
        "black high-waist athletic pants with thin gray side stripe, white running shoes, "
        "black hair in high ponytail"),
    2: ("medium shot, the athletic woman doing light stretching at gym mat, "
        "warm-up motion, bright open gym with equipment in background, "
        "wearing misty-blue long-sleeve athletic top, black high-waist athletic pants, "
        "white running shoes, high ponytail"),
    3: ("full body shot, the athletic woman performing bodyweight squats at squat rack, "
        "strong form, looking at camera, bright gym floor with mirror walls, "
        "wearing misty-blue long-sleeve athletic top with white arc logo on left chest, "
        "black high-waist athletic pants with thin gray side stripe, "
        "white running shoes, high ponytail"),
    4: ("medium shot, the athletic woman doing cool-down stretching, "
        "touching her toes, calm expression, bright gym, "
        "wearing misty-blue long-sleeve athletic top with white arc logo on left chest, "
        "black high-waist athletic pants with thin gray side stripe, "
        "white running shoes, high ponytail"),
    5: ("medium shot, the athletic woman taking a mirror selfie with phone at gym, "
        "showing a thumbs up pose, bright gym mirror reflection visible, "
        "wearing misty-blue long-sleeve athletic top with white arc logo on left chest, "
        "black high-waist athletic pants with thin gray side stripe, "
        "white running shoes, high ponytail"),
}

SHOTS_BY_NAME = {"xiaoya": SHOTS_XIAOYA, "linxia": SHOTS_LINXIA}

# Back-compat: 默认 SHOTS = xiaoya
SHOTS = SHOTS_XIAOYA

# 小雅台词
LINES_XIAOYA = {
    1: "今天又是元气满满的一天, 加油打工人!",
    2: "这次的项目方案大家看看, 有问题随时提。",
    3: "午餐时间到, 今天吃什么呢?",
    4: "这份报告下周要给客户, 大家辛苦一下。",
    5: "下班啦, 明天见!",
}

# 林夏台词 (健身房钩子 + 反转"先拍照", 复用 ep03 的反转叙事节奏)
LINES_LINXIA = {
    1: "装备先到位, 健身才有味儿!",
    2: "热身做好了, 才不容易受伤哦。",
    3: "深蹲 12 个一组, 我先做 3 组!",
    4: "训练完了不拉伸, 等于白练啦!",
    5: "等等, 先拍照! 先拍照发个朋友圈!",
}

LINES_BY_NAME = {"xiaoya": LINES_XIAOYA, "linxia": LINES_LINXIA}

# i2v 运动 (Seedance 1.0 pro; 低幅度保衣服一致)
I2V_MOTION_XIAOYA = {
    1: ("the slim fit woman stands calmly at the office reception desk, gentle posture, "
        "subtle breathing motion, "
        "her face stays sharp and front-facing, minimal head movement, "
        "the white office blouse stays white throughout, the pencil skirt stays the same throughout"),
    2: ("the slim fit woman sits at the conference table gesturing with one hand, "
        "subtle presentation motion, confident expression, "
        "her face stays sharp and stable, minimal head movement, "
        "the white office blouse stays white throughout, the pencil skirt stays the same throughout"),
    3: ("the slim fit woman holds chopsticks and lifts a small bite, "
        "subtle eating motion, smiling, "
        "her face stays sharp and front-facing, minimal head movement, "
        "the white office blouse stays white throughout, the pencil skirt stays the same throughout"),
    4: ("the slim fit woman holds up report papers, looking at camera, "
        "subtle speaking motion, "
        "her face stays sharp and stable, minimal head movement, "
        "the white office blouse stays white throughout, the pencil skirt stays the same throughout"),
    5: ("the slim fit woman picks up her handbag and waves one hand, "
        "subtle goodbye motion, cheerful expression, "
        "her face stays sharp and front-facing, minimal head movement, "
        "the white office blouse stays white throughout, the pencil skirt stays the same throughout"),
}

# 林夏 i2v motion — 加 3 句不变式锁衣服 (ep03 实战教训: 雾蓝长袖+白弧+黑高腰+灰侧线+白鞋+高马尾)
I2V_MOTION_LINXIA = {
    1: ("the athletic woman walks into a bright gym, casual confident stride, "
        "subtle breathing motion, "
        "her face stays sharp and front-facing, minimal head movement, "
        "the misty-blue athletic top stays misty-blue throughout, "
        "the white arc logo on the left chest stays visible throughout, "
        "the black high-waist athletic pants stay black throughout, "
        "the thin gray side stripe stays visible throughout, "
        "the white running shoes stay white throughout, the high ponytail stays high throughout"),
    2: ("the athletic woman does gentle warm-up stretches, light motion, "
        "looking at camera, "
        "her face stays sharp and stable, minimal head movement, "
        "the misty-blue athletic top stays misty-blue throughout, "
        "the white arc logo on the left chest stays visible throughout, "
        "the black high-waist athletic pants stay black throughout, "
        "the thin gray side stripe stays visible throughout, "
        "the white running shoes stay white throughout, the high ponytail stays high throughout"),
    3: ("the athletic woman performs bodyweight squats, strong controlled form, "
        "minimal head movement, "
        "her face stays sharp and front-facing, "
        "the misty-blue athletic top stays misty-blue throughout, "
        "the white arc logo on the left chest stays visible throughout, "
        "the black high-waist athletic pants stay black throughout, "
        "the thin gray side stripe stays visible throughout, "
        "the white running shoes stay white throughout, the high ponytail stays high throughout"),
    4: ("the athletic woman does cool-down stretching, touching her toes, "
        "calm expression, "
        "her face stays sharp and stable, minimal head movement, "
        "the misty-blue athletic top stays misty-blue throughout, "
        "the white arc logo on the left chest stays visible throughout, "
        "the black high-waist athletic pants stay black throughout, "
        "the thin gray side stripe stays visible throughout, "
        "the white running shoes stay white throughout, the high ponytail stays high throughout"),
    5: ("the athletic woman holds up her phone taking a mirror selfie, "
        "showing a thumbs up, "
        "her face stays sharp and front-facing, minimal head movement, "
        "the misty-blue athletic top stays misty-blue throughout, "
        "the white arc logo on the left chest stays visible throughout, "
        "the black high-waist athletic pants stay black throughout, "
        "the thin gray side stripe stays visible throughout, "
        "the white running shoes stay white throughout, the high ponytail stays high throughout"),
}

I2V_MOTION_BY_NAME = {"xiaoya": I2V_MOTION_XIAOYA, "linxia": I2V_MOTION_LINXIA}

# SPEAKER 路由 (单角色 = 全本角色)
SPEAKER_XIAOYA = {n: "XIAOYA" for n in range(1, 6)}
SPEAKER_LINXIA = {n: "LINXIA" for n in range(1, 6)}
SPEAKER_BY_NAME = {"xiaoya": SPEAKER_XIAOYA, "linxia": SPEAKER_LINXIA}

# Back-compat
I2V_MOTION = I2V_MOTION_XIAOYA
SPEAKER = SPEAKER_XIAOYA
LINES = LINES_XIAOYA