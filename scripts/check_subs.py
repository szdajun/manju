"""客观验证: 10 镜字幕换行后每行像素宽度是否 <= 画布安全宽(720 - 2*pad)。

不靠 vision(这批 CDN URL 含反斜杠, analyze_image 不可靠), 走像素量。
"""
import sys
import os

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MANJU, "scripts"))

from make_ep01 import wrap_text, SHOTS, W, FONT  # noqa: E402
from PIL import ImageDraw, ImageFont, Image  # noqa: E402

PAD_X = 30
SAFE = W - 2 * PAD_X
font = ImageFont.truetype(FONT, 42)
probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

print(f"canvas W={W}  safe_text_width <= {SAFE}\n")
print(f"{'shot':>4} {'lines':>5} {'maxw_px':>8}  {'fits':>5}  wrapped")
print("-" * 70)
allok = True
for shot, dur, kind, src, line, kb in SHOTS:
    ls = wrap_text(line)
    ws = [int(probe.textlength(l, font=font)) for l in ls]
    maxw = max(ws)
    fits = maxw <= SAFE
    allok = allok and fits
    tag = "OK" if fits else "CUT"
    print(f"{shot:>4} {len(ls):>5} {maxw:>8}  {tag:>5}  {ls}")
print("-" * 70)
print("ALL FITS:", allok)
sys.exit(0 if allok else 1)
