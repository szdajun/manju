"""客观闭环: 实际渲染 10 张字幕 PNG, 像素级断言无边缘裁切。

check_subs 验的是 wrap 逻辑(行宽<=安全宽); 本脚本验的是 render_subtitle 真实输出 ——
扫 RGBA, 若有任何不透明像素(alpha>0)落在左右边缘 5px 内, 即判定裁切。比 vision 可靠。
"""
import os
import sys
import tempfile

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MANJU, "scripts"))

from make_ep01 import render_subtitle, SHOTS, W  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

EDGE = 5  # 左右各 5px 内不得有不透明像素
tmp = tempfile.mkdtemp(prefix="subchk_")
print(f"canvas W={W}  边缘禁入区: x<{EDGE} 或 x>{W-1-EDGE}\n")
print(f"{'shot':>4} {'minx':>5} {'maxx':>5} {'alphamax':>9}  {'clip':>5}")
print("-" * 45)
allok = True
for shot, dur, kind, src, line, kb in SHOTS:
    p = os.path.join(tmp, f"s{shot:02d}.png")
    render_subtitle(line, p)
    arr = np.array(Image.open(p))          # (H, W, 4) RGBA
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    minx, maxx = int(xs.min()), int(xs.max())
    amax = int(alpha.max())
    clip = minx < EDGE or maxx > (W - 1 - EDGE)
    allok = allok and not clip
    print(f"{shot:>4} {minx:>5} {maxx:>5} {amax:>9}  {'CLIP' if clip else 'ok':>5}")
print("-" * 45)
print("NO EDGE CLIP:", allok)
sys.exit(0 if allok else 1)
