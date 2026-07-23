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

# 婷婷角色指纹(配角, 2026-07-22 新增立绘 + 双锁脸用)。**必须与小雅可区分**,
# 否则双人镜双 IPAdapter 会把两张脸锁混成一团。婷婷 = 更成熟、深棕波浪发、浅蓝上衣的同事。
# ⚠ **不戴眼镜**(2026-07-22: 初版戴眼镜, 双锁时眼镜框 bleed 到小雅脸上(半框 artifact);
#   改用发色/发型(深棕波浪 vs 小雅长直黑)+ 年龄 + 脸型 区分, 避 bleed)。
TINGTING = ("1girl, solo, young Chinese woman in her late 20s, slightly more mature friendly face, "
            "softer rounder face contour, shoulder-length wavy dark brown hair with side bangs, "
            "light blue office blouse with a thin cardigan, "
            "average build, warm curious gossip-loving expression, office attire")

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
    2: "小雅！你最近是不是瘦了？身材变好好！整个人精神多了！",  # Phase2 轻加长(22字)+降速(1.35→~1.1) 占满 5.04s 镜头, 消灭静音尾→ MuseTalk 全程音频驱动, 不再冻嘴/变形; 文案+语速结合(不只靠文案), 2026-07-23
    3: "有吗？还好啦～可能是最近早睡早起，作息比以前更规律了吧！",  # Phase2 轻加长(24字)@speed1.1≈5.5s填满5.04s镜头, 2026-07-23
    4: "快老实交代——你最近是不是背着我，偷偷吃了什么减肥药啊？",  # Phase2 轻加长(24字)@speed1.1, 2026-07-23
    5: "才没有！我才不碰那些东西呢，减肥药多伤身体啊，我可不傻！",  # Phase2 轻加长(24字)@speed1.1, 2026-07-23
    6: "那是……抽脂？代餐？断碳水？到底是什么呀？",  # Phase2 加长(15字)@1.0≈5.06s填满5.04s — shot6 i2v意外带脸(剧本"无脸"假设作废), 需补MuseTalk唇同步; 保留原"抽脂/代餐/断碳水"短促猜+追问尾巴, 21字@1.0=7.08s过头, 2026-07-23
    7: "都不是！我每天下班，就去时代广场，跟着大家一起跳操啦！",  # Phase2 轻加长(23字)@speed1.1, 2026-07-23
    8: "音乐一起，跟着领操一起跳，可带劲了！",  # 旁白(voice_ep02 SPEAKER[8]=NARRATOR 已路由, 前缀不入字幕/音频)
    9: "真的假的？光每天去跳个操而已，就能瘦这么多？我才不信呢！",  # Phase2 轻加长(24字)@speed1.1, 2026-07-23
    10: "那我也要去！今晚下班就一起去！",
    # 收尾品牌 tag(2026-07-23 用户追加「我们是打工牛马，健身达人！」)。一人一句绕开 MuseTalk 单脸限制:
    # shot11=小雅说上句, shot12=婷婷接下句; 动作"曲臂握拳"=power pose(见 SHOTS_REAL_OVERRIDE[11/12])。
    # 配音单独合成 voice_shot11/12.mp3(非 voice_ep02 批量; voice_tag11_xiaoya/voice_tag12_tingting 拷贝来)。
    11: "我们是打工牛马，",
    12: "健身达人！",
}

# i2v 运动意图(LTX; face-stable 后缀在 gen_ep02_i2v 外部拼, 这里只写运动)
I2V_MOTION = {
    1: "the slim fit woman walks calmly past the desks holding a folder, the colleague slowly looks up, "
       "subtle motion, the slim fit woman's face stays sharp and in focus",
    2: "the colleague slowly stands up examining the slim fit woman, subtle motion, gentle movement",
    # shot3 去捂嘴(2026-07-23, 同 SHOTS_REAL_OVERRIDE[3] 改动): 运动也去捂嘴, 显式约束手远离脸/嘴,
    # 给 Wan i2v 一个嘴全程可见的首帧+运动, Phase2 MuseTalk 才能干净对口型.
    3: "the slim fit woman tilts her head with a growing shy proud smile, one hand gently tucking hair "
       "behind her ear, face stays sharp and front-facing, subtle gentle motion, both hands stay down "
       "and away from her face and mouth the entire time",
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

# ===== 正反打写实板子覆盖 (2026-07-23 治互动感 + 给 Phase2 口型靶子) =====
# 用户反馈"两角色像模特摆谱, 缺互动感"。根因: 婷婷的台词(2/4/9)原本把她画成侧/背/身后,
# 说话时看不到正脸 → 既没互动(各摆各的)也没法对口型。改成"婷婷 solo 正面说话、视线偏向画外
# (看画外的小雅)" → 眼神线匹配(婷婷看右 → 切小雅)建立互动(靠剪辑, 不靠 i2v 硬生成两人配合);
# 且单正脸利于锁脸 + Phase2 MuseTalk 对口型。
# 仅写实版(gen_ep02_real_batch 关键帧 + gen_ep02_seedance i2v)读; 漫画版仍走原 SHOTS(小雅主体)。
SHOTS_REAL_OVERRIDE = {
    2: ("medium close-up, the colleague stands facing the camera front-facing, looking off to the side "
        "with mouth open in amazed surprise, both hands gesturing excitedly, her face clearly visible "
        "and expressive, bright modern office"),
    # shot3 去捂嘴(2026-07-23): 原 SHOTS[3]="covering her mouth with one hand" → 手物理盖嘴,
    # MuseTalk 唇同步画在手/手缝上=崩, 不上唇同步=嘴不动(用户两轮报). 改"侧头浅笑+撩耳侧头发露脸"
    # 的嘴可见害羞, 既留害羞人设又给 Phase2 口型干净靶子. 仅写实版(gen_ep02_real_batch)读;
    # 漫画版 SHOTS[3] 捂嘴不动(已冻结). 台词 LINES[3] 不变 → TTS 音频复用.
    3: ("medium close-up, the slim fit woman stops and turns around, tilting her head slightly with a "
        "shy proud smile, one hand lightly tucking hair behind her ear, mouth clearly visible and "
        "relaxed, both hands kept down and away from her face, face clearly visible front facing, office"),
    4: ("close-up, the colleague facing the camera front-facing, leaning in with a gossipy knowing smirk, "
        "one hand near her mouth whispering, eyes looking off to the side playfully, "
        "her face clearly visible, office"),
    # shot6 改婷婷追问镜(2026-07-23 修复角色错位): 原走 else 分支锁小雅, 但 LINES[6]
    # 「抽脂/代餐/断碳水到底是什么呀」是婷婷追问台词 → 画面小雅+台词婷婷=角色错位(用户报)。
    # 改婷婷正面追问(同 shot4/9 正反打风格) + 保留"数手指列举"原 shot6 元素(但是婷婷的手+脸)
    # + 嘴张给 MuseTalk 干净靶 + 视线偏画外(看小雅)建互动。
    6: ("medium close-up, the colleague faces the camera front-facing, leaning in with eager nosy curiosity, "
        "one hand gesturing counting on fingers one by one, mouth open mid-question, eyes wide with gossip, "
        "looking off to the side, her face clearly visible, bright modern office"),
    9: ("medium shot, the colleague faces the camera front-facing, looking off to the side with eyes wide "
        "in excited disbelief, clenching one fist in front of her chest, mouth open, "
        "her face clearly visible, office"),
    # shot11/12 = 收尾品牌 tag「我们是打工牛马，健身达人！」(2026-07-23 用户追加)。
    # 一人说一句绕开 MuseTalk 单脸限制: shot11=小雅(主角)说上句, shot12=婷婷(接梗)说下句。
    # 用户指定动作"曲臂握拳"= bent-arm clenched-fist power pose, 正扣"健身达人"力量感。
    # 正面对镜头 + 嘴可见(给 MuseTalk 干净口型靶) + office(与全片场景一致)。
    # ⚠ 握拳重写(2026-07-23): 初版抽象措辞"bending one arm with a clenched fist"被 SD1.5 无视
    #   (两镜都没出拳)。SD1.5 手/拳是弱项, 改具体空间锚点("raises one arm bent at the elbow up to
    #   shoulder height, hand clenched into a tight fist held up near the shoulder")+ (…:1.3) 强调 +
    #   多 seed 重抽挑真握拳帧。脚本次 char 描述自动前置(CHAR/TINGTING), 这里只写 pose+scene。
    11: ("medium shot, facing the camera front-facing with a confident proud smile, "
         "raising one arm bent at the elbow up to shoulder height, hand clenched into a tight fist "
         "held up near the shoulder, flexing in a strong motivated power pose, "
         "(arm bent at elbow with raised clenched fist:1.3), knuckles visible, well drawn hand, "
         "energetic fitness attitude, mouth clearly visible, face clearly visible front facing, "
         "bright modern office"),
    12: ("medium shot, facing the camera front-facing with an excited cheerful smile, "
         "raising one arm bent at the elbow up to shoulder height, hand clenched into a tight fist "
         "held up near the shoulder, flexing in a strong motivated power pose, "
         "(arm bent at elbow with raised clenched fist:1.3), knuckles visible, well drawn hand, "
         "energetic fitness attitude, mouth clearly visible, face clearly visible front facing, "
         "bright modern office"),
}
I2V_MOTION_REAL_OVERRIDE = {
    2: "the colleague turns to face the camera speaking with amazed surprise, looking off to the side, "
       "gesturing with both hands, subtle lively motion, her face stays sharp and expressive",
    4: "the colleague leans slightly toward the camera whispering with a gossipy smirk, looking off to the side, "
       "subtle motion, her face stays front-facing and sharp",
    # shot6 婷婷追问(2026-07-23 角色错位修复, 见 SHOTS_REAL_OVERRIDE[6])
    6: "the colleague leans in eagerly, gesturing with her hand counting on fingers one by one, "
       "mouth moving mid-question, eyes wide with nosy curiosity, looking off to the side, "
       "subtle lively motion, her face stays sharp and front-facing",
    9: "the colleague clenches her fist with growing excited disbelief, eyes wide, looking off to the side, "
       "subtle lively motion, her face stays sharp and expressive",
    # shot11/12 = 收尾 tag「我们是打工牛马，健身达人！」曲臂握拳(2026-07-23)。
    # SD1.5 关键帧做不出拳(8 次 0 拳, 能力天花板), 赌 Seedance i2v 视频先验比 SD1.5 图像先验强 →
    # 用"抬臂+攥拳上举"动态意图在动画里把拳攥出来(=用户要的"真动作")。脸锚 FACE_POS 由
    # gen_ep02_seedance 自动拼(11/12 不在 NO_FACE_SHOTS), 保脸稳给 Phase2 MuseTalk 干净靶。
    11: ("the slim fit woman dynamically raises one arm bent at the elbow upward to shoulder height, "
         "her hand clenching into a tight clenched fist that she pumps upward in a strong motivated power pose, "
         "confident energetic fitness attitude, knuckles visible, the fist is clearly clenched and raised high, "
         "her face stays sharp stable front-facing and centered, minimal head movement, both eyes open and natural, "
         "mouth clearly visible and relaxed, subtle breathing, bright modern office"),
    12: ("the colleague dynamically raises one arm bent at the elbow upward to shoulder height, "
         "her hand clenching into a tight clenched fist that she pumps upward in a strong motivated power pose, "
         "excited energetic fitness attitude, knuckles visible, the fist is clearly clenched and raised high, "
         "her face stays sharp stable front-facing and centered, minimal head movement, both eyes open and natural, "
         "mouth clearly visible and relaxed, subtle breathing, bright modern office"),
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
