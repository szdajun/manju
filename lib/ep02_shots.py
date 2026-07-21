"""ep02《她到底怎么瘦的?》分镜剧本(办公室两女 → 时代广场跳操 CTA)。漫画版先行。

叙事弧: 钩子(身材变化) → 悬念(猜减肥药) → 反转(时代广场跳操) → CTA(一起去)。
角色: 小雅(主角, 变瘦变美 fit toned, IPAdapter 全程锁脸同人, 脸 dominant 保 face_sim 主脸选对)
  + 婷婷(配角, **不锁脸**, 构图用 side profile / back / hands / 同框次要脸 规避 → 省立绘降脸漂)。
风格: 漫画先行(japanese manga style, 复用 gen_shot 的 STYLE/NEG 反逼写实); 写实版回头完善(WSL+Wan)。
i2v: LTX-Video 本地(漫画 LTX 历史 face_sim 持 0.67-0.70, 关键帧紧 0.64-0.75)。
联动: shot8 时代广场闪回 = LTX 漫剧风(画风统一), 呼应 fitness 主管线 bg_swap 时代广场主题。

- CHAR/STYLE/NEG 供 gen_ep02_batch 关键帧(IPAdapter 锁小雅立绘)。
- SHOTS 关键帧静态场景; LINES 中文字幕/旁白; I2V_MOTION LTX 运动意图(face-stable 后缀外部拼)。
- shot6 纯手部特写无主角脸 → face_sim 门在 batch 脚本对该镜 skip(不测)。
"""

# 小雅角色指纹(每镜带, 锁服装/发型/身材; 配合 IPAdapter 锁脸)。fit toned = 健身后好身材
CHAR = ("1girl, solo, slim fit young Chinese woman, fit toned body, long straight black hair, "
        "bright cheerful face, confident smile, white office blouse, pencil skirt, office attire")

# 漫画风格(复用 gen_shot.STYLE, 反逼写实)
STYLE = ("japanese manga style, thick black lineart, flat shading, high contrast, "
         "simple background, humorous, sweat drops, comic, vertical composition")

# 关键帧静态场景(小雅脸 dominant; 婷婷用 colleague/side/hands 非 frontal 省立绘)
SHOTS = {
    1: ("medium shot, bright modern office, the slim fit woman walks past desks holding a folder, "
        "a female colleague seated at a desk looks up with a surprised impressed expression, daytime office"),
    2: ("medium shot, the slim fit woman stands confidently in front, her female colleague stands up "
        "behind her looking amazed examining her figure, office"),
    3: ("medium close-up, the slim fit woman stops and turns around, covering her mouth with one hand, "
        "shy proud smile, face clearly visible front facing, office"),
    4: ("medium shot, the slim fit woman listening, her colleague leans in from the side in 3/4 profile "
        "with a gossipy knowing smirk, whispering, office"),
    5: ("medium shot, the slim fit woman waving both hands firmly, shaking her head in denial, "
        "face clearly visible, office"),
    6: ("close-up of a woman's hand counting on fingers one by one, blurred office background, no face"),
    7: ("medium shot, the slim fit woman proudly holds up her smartphone showing a video screen, "
        "confident smile, face clearly visible front facing, office"),
    8: ("medium shot, dancing energetically doing aerobics with arms raised, bright city square at night "
        "with colorful neon signs, a couple of people dancing along behind, lively festive atmosphere"),
    9: ("medium shot, the slim fit woman smiling in front, her colleague beside her clenching a fist "
        "with excitement eyes wide, office"),
    10: ("medium shot, the slim fit woman and her colleague linking arms smiling at each other happily, "
         "the slim fit woman's face clearly visible, office"),
}

# 台词(中文字幕/旁白)
LINES = {
    1: "最近小雅……好像变了一个人？",  # 旁白(voice_ep02 SPEAKER[1]=NARRATOR 已路由, 前缀不入字幕/音频)
    2: "小雅！你最近是不是瘦了？身材变好好！",
    3: "有吗？还好啦～",
    4: "快老实交代——是不是吃了什么减肥药？",
    5: "才没有！吃药多伤身体啊。",
    6: "那是……抽脂？代餐？断碳水？",
    7: "都不是！我每天下班，去时代广场跳操！",
    8: "音乐一起，跟着领操一起跳，可带劲了！",  # 旁白(voice_ep02 SPEAKER[8]=NARRATOR 已路由, 前缀不入字幕/音频)
    9: "真的假的？跳操能瘦这么多？！",
    10: "那我也要去！今晚下班就一起去！",
}

# i2v 运动意图(LTX; face-stable 后缀在 gen_ep02_i2v 外部拼, 这里只写运动)
I2V_MOTION = {
    1: "the slim fit woman walks calmly past the desks holding a folder, the colleague slowly looks up, "
       "subtle motion, the slim fit woman's face stays sharp and in focus",
    2: "the colleague slowly stands up examining the slim fit woman, subtle motion, gentle movement",
    3: "the slim fit woman raises one hand to cover her mouth with a growing smile, "
       "face stays sharp, gentle motion",
    4: "the colleague leans in slightly whispering with a smirk, both faces stay front-facing, slow subtle motion",
    5: "the slim fit woman waves both hands side to side shaking her head, small firm denial gestures, "
       "face stays sharp and stable",
    6: "a woman's hand slowly counts on fingers one by one, small hand motion, no face",
    7: "the slim fit woman raises her smartphone showing the screen, confident smile, small movement, "
       "face stays stable",
    8: "energetic aerobics dance with arm movements and gentle bouncing, neon lights flickering, "
       "the people behind sway gently, face stays clear and in focus",
    9: "the colleague clenches a fist with growing excitement, subtle motion, the slim fit woman smiling",
    10: "the two women stand side by side linking arms, both facing the camera with warm happy smiles, very subtle gentle motion and soft breathing, both faces stay sharp symmetric and well-formed throughout, both women's eyes stay open sharp symmetric and natural for the entire clip with no distortion or asymmetry, both women have natural well-formed eyelids with eyes normally open, no drooping or squinting eyelids, the slim fit woman's face and eyes stay perfectly still front-facing, the colleague's face also stays still sharp and natural, no head turning, no leaning toward each other, no face distortion, no eye distortion",
}

# 漫画 NEG(反逼写实, 复用 gen_shot.NEG_DEFAULT; + 脸形变项治 i2v 脸漂)
NEG = ("text, watermark, logo, signature, extra fingers, deformed hands, mutated hands, "
       "bad anatomy, extra limbs, blurry, realistic, photorealistic, 3d, photograph, "
       "lowres, jpeg artifacts, multiple views, reference sheet, "
       "face morphing, face distortion, warped face, melting face, asymmetrical eyes, deformed face")

# ===== 写实版(gen_ep02_real_batch 用; WSL+Wan2.2-5B i2v 写实路径, 2026-07-21) =====
# 写实风格(反逼漫画, 与上 manga STYLE 对立; 复用 gen_ep01_real_batch 写实套路)。CHAR 已是写实描述直接复用。
REAL_STYLE = ("photorealistic, real photograph, 50mm lens, cinematic lighting, shallow depth of field, "
              "detailed skin texture, natural colors, sharp focus, vertical composition, 8k uhd")
# 写实 NEG: 砍 "realistic/photorealistic/3d/photograph"(那是 manga 版反逼漫画用的), 加反漫画/反 CG + 脸形变项
#   + 反厚下眼睑(2026-07-21 用户审片: 下眼帘偏厚显疲惫, 淡化合小雅精神人设)
REAL_NEG = ("text, watermark, logo, signature, extra fingers, deformed hands, mutated hands, "
            "bad anatomy, extra limbs, blurry, lowres, jpeg artifacts, "
            "anime, manga, cartoon, illustration, painting, 3d render, cg, lineart, flat shading, "
            "plastic skin, airbrushed, multiple views, reference sheet, "
            "face morphing, face distortion, warped face, melting face, asymmetrical eyes, deformed face, "
            "puffy under-eye, heavy eye bags, thick lower eyelid, prominent eye bags, dark circles, tired eyes")
