"""Generate cartoon placeholder ref.png for kid_dance LoRA training.

This is a POC placeholder. The user can replace with a proper SDXL/手绘 ref
image later — see .superpowers/sdd/task-1-brief.md Step 1 path (b)/(c).
Style matches project convention (cartoon, comic book outlines, big head
on small body 1:3) used in sibling 立绘 assets.
"""
from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
W, H = SIZE, SIZE  # square for LoRA training portability

img = Image.new("RGB", (W, H), (250, 240, 230))  # soft pastel background
d = ImageDraw.Draw(img)

# Background soft circle halo
d.ellipse((100, 80, W - 100, H - 80), fill=(255, 218, 224))  # pastel pink halo

# 1:3 head:body proportions
# Head (big) center upper
head_cx = W // 2
head_cy = int(H * 0.32)
head_r = int(W * 0.18)

# Hair (back) — black bob
d.ellipse((head_cx - head_r - 18, head_cy - head_r - 22,
           head_cx + head_r + 18, head_cy + head_r + 8), fill=(30, 22, 18))
# Face
d.ellipse((head_cx - head_r, head_cy - head_r,
           head_cx + head_r, head_cy + head_r), fill=(255, 222, 198))
# Cheek blush
d.ellipse((head_cx - head_r + 20, head_cy + 20,
           head_cx - head_r + 75, head_cy + 55), fill=(255, 168, 168))
d.ellipse((head_cx + head_r - 75, head_cy + 20,
           head_cx + head_r - 20, head_cy + 55), fill=(255, 168, 168))
# Bangs
d.chord((head_cx - head_r - 5, head_cy - head_r - 12,
         head_cx + head_r + 5, head_cy + 10), start=180, end=360, fill=(30, 22, 18))
# Eyes (confident almond shape, asian-looking)
for ex in (head_cx - 55, head_cx + 55):
    # eye whites
    d.ellipse((ex - 22, head_cy - 12, ex + 22, head_cy + 12), fill=(255, 255, 255))
    # iris dark
    d.ellipse((ex - 14, head_cy - 10, ex + 14, head_cy + 12), fill=(40, 28, 22))
    # pupil highlight
    d.ellipse((ex - 6, head_cy - 6, ex + 6, head_cy + 4), fill=(255, 255, 255))
    # eyelash upper
    d.arc((ex - 24, head_cy - 16, ex + 24, head_cy + 14), start=190, end=350, fill=(0, 0, 0), width=4)
# Confident smirk (slight asymmetric lift, mouth right side raised)
d.arc((head_cx - 40, head_cy + 50, head_cx + 50, head_cy + 95), start=15, end=165, fill=(180, 50, 60), width=6)
# Tiny nose hint
d.arc((head_cx - 8, head_cy + 25, head_cx + 10, head_cy + 40), start=180, end=360, fill=(220, 170, 150), width=3)

# Body (small, 1:3) — red sleeveless top
body_top_y = head_cy + head_r + 5
body_top_w = int(W * 0.28)
body_bot_y = int(H * 0.78)
# Neck
d.rectangle((head_cx - 22, head_cy + head_r - 12, head_cx + 22, body_top_y + 18), fill=(255, 222, 198))
# Top trapezoid (broaden at hip)
top_pts = [
    (head_cx - 90, body_top_y + 18),
    (head_cx + 90, body_top_y + 18),
    (head_cx + body_top_w, body_bot_y - 140),
    (head_cx - body_top_w, body_bot_y - 140),
]
d.polygon(top_pts, fill=(220, 50, 60))  # vivid red sleeveless top
# Top outline
d.line(top_pts + [top_pts[0]], fill=(0, 0, 0), width=3)
# White loose pants
pants_top_y = body_bot_y - 140
pants_pts_l = [
    (head_cx - body_top_w, pants_top_y),
    (head_cx - body_top_w + 30, H - 130),
    (head_cx - 28, H - 130),
    (head_cx - 28, pants_top_y),
]
pants_pts_r = [
    (head_cx + 28, pants_top_y),
    (head_cx + 28, H - 130),
    (head_cx + body_top_w - 30, H - 130),
    (head_cx + body_top_w, pants_top_y),
]
d.polygon(pants_pts_l, fill=(245, 245, 240))  # white
d.polygon(pants_pts_r, fill=(245, 245, 240))
d.line(pants_pts_l + [pants_pts_l[0]], fill=(0, 0, 0), width=3)
d.line(pants_pts_r + [pants_pts_r[0]], fill=(0, 0, 0), width=3)
# White sneakers (simple ovals)
d.ellipse((head_cx - body_top_w + 10, H - 135, head_cx - 10, H - 90), fill=(255, 255, 255), outline=(0, 0, 0), width=3)
d.ellipse((head_cx + 10, H - 135, head_cx + body_top_w - 10, H - 90), fill=(255, 255, 255), outline=(0, 0, 0), width=3)
# Arms (small, raised for dance vibe — one up, one out)
import math

def arm(side):
    sign = 1 if side == "right" else -1
    sx = head_cx + sign * 95
    sy = body_top_y + 35
    # bent upward
    ex = head_cx + sign * 175
    ey = body_top_y - 60 if side == "left" else body_top_y + 110
    d.line([(sx, sy), (ex, ey)], fill=(255, 222, 198), width=42)
    # hand ball
    d.ellipse((ex - 24, ey - 24, ex + 24, ey + 24), fill=(255, 222, 198), outline=(0, 0, 0), width=3)
arm("left")
arm("right")

# Slight motion line sticker — kid dancing
# Tiny sparkle confetti
for cx, cy, c in [(W*0.18, H*0.30, (255,210,80)), (W*0.82, H*0.40, (120,200,255)),
                  (W*0.20, H*0.55, (200,150,255)), (W*0.80, H*0.60, (255,150,200))]:
    d.ellipse((cx-8, cy-8, cx+8, cy+8), fill=c)

# Soften (slight blur for cartoon softness)
img = img.filter(ImageFilter.GaussianBlur(0.6))

out = "assets/characters/kid_dance/ref.png"
img.save(out, "PNG", optimize=True)
print(f"wrote {out} ({img.size}, mode={img.mode})")
