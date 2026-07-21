"""Wan2.2-5B I2V 逐镜运动意图 prompt(单模型, 无 14B/1.3B 分工)。

2026-07-20 拍板: 改 Wan2.2-TI2V-5B 单模型全 10 镜, 删 SHOTS_14B/SHOTS_13B/MODEL_FOR_SHOT。
prompt 描述"要发生的连续运动"(非静态), 供 WanVideoTextEncode.positive_prompt。
"""

# 运动意图(写真版): 描述要发生的运动, 非静态构图
WAN_PROMPTS = {
    1: "the man slowly blinks and lets out a tired sigh, slight head droop, "
       "blue computer screen glow flickering on his face, mostly subtle motion",
    2: "the man slumps further down in the office chair, head tilting, exhausted breathing",
    3: "the man walks hesitantly into the bright gym, looking around nervously, "
       "bodybuilders lifting weights in the background",
    4: "the man strains and trembles trying to lift a small dumbbell with both hands, "
       "face turning red with effort, arms shaking, background bodybuilder curling easily",
    5: "the man on a treadmill slowly loses balance and stumbles backwards, "
       "comedic slow-motion fall, legs giving out, face stays visible",
    6: "the man pours protein powder into a mug, stirs and takes a sip, "
       "whole face scrunching up in disgust, shaking his head",
    7: "the man doing a plank, whole body trembling and shaking violently, "
       "a fly lands on his nose, frantic unsteady motion",
    8: "the man lying flat on the gym floor weakly raises a thumbs up, "
       "trainer stands still looking down at him",
    9: "the man looks at his smartphone, expression shifting from despair to relief, slight head shake",
    10: "the man sits at his desk with a sports sweatband, picks up a small dumbbell, "
        "determined yet helpless eyes, slow deliberate motion",
}

WAN_NEG = ("text, watermark, logo, distorted face, deformed hands, morphing identity, "
           "extra fingers, bad anatomy, blurry, low quality, violent scene change, "
           "mouth disappearing, melting face, face morphing, cartoon, anime, "
           "distorted eyes, asymmetric eyes, uneven eyes, melting eyes, cross-eyed, "
           "lazy eye, misshapen irises, warped eyes, mismatched eyes, "
           "drooping eyelids, ptosis, hooded eyelids, heavy eyelids, puffy eyelids, "
           "uneven eyelids, lower eyelid too high, squinting, sanpakugan, sclera showing, "
           "bags under eyes, bloodshot eyes")
