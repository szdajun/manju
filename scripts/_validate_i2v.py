"""临时: 1 条 Seedance i2v 验证(写实立绘 #3 → 视频)。

验三件 (用户选项 A "搭写实立绘→验1条"):
  1. 真人脸政策: 1.0 pro 无真人脸禁令 + 本图全合成 → 应放行(确认无 policy error)
  2. face_sim命门: 立绘 vs i2v 中段帧 arcface cosine (≥0.45 才算脸撑住运镜)
  3. 写实质感: 主观, vision 不可用 → 留用户回看 (本脚本只产片)

prompt 刻意"极轻运镜 + 脸全程正脸可见", 把 face_sim 通过率拉满。
用完即删。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.video_gen import seedance_i2v  # noqa: E402

IMG = "../manju/output/lihui_real/主角立绘写实.png"
OUT = "../manju/output/ep01_i2v_real"
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "validate_lihua.mp4")

assert os.path.exists(IMG), f"缺立绘: {IMG}"

PROMPT = (
    "a tired young male office worker slowly turns his head to look slightly to the side, "
    "subtle natural breathing, heavy tired eyes with dark circles, messy short hair, "
    "light blue plaid shirt, static camera, soft cinematic indoor lighting, photorealistic, "
    "face clearly visible throughout"
)

print(f"[validate] img={IMG}", flush=True)
print(f"[validate] out={out} (480p/5s/9:16, ~49005 tokens)", flush=True)
seedance_i2v(IMG, PROMPT, out, duration=5, ratio="9:16", resolution="480p")
print(f"\n[validate] DONE -> {out}", flush=True)
