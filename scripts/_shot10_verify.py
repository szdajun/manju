"""shot10 双人镜重抽客观验脸 (2026-07-22)。

face_sim 命门只测主脸(小雅)身份, 测不到: ① 局部眼睛瑕疵(用户肉眼判) ② 配角(婷婷)漂回。
本脚本补 ② 并复述 ① 之外的硬指标: 按"vs 小雅 REF 余弦"分身份 — max=小雅(锁脸),
min=婷婷(配角)。然后:
  - 小雅 sim_vs_REF 逐帧 (身份锁, 要 ≥0.45)
  - 婷婷 sim vs 关键帧婷婷源脸 (漂离源, 要 ≥0.85)
  - 婷婷 逐帧自稳定余弦 (要 ≥0.85)
身份法比位置法(bbox x 分左右)稳: 不依赖两人空间不互换。跑在 fitness venv。

用法: FIT_PY scripts/_shot10_verify.py <mp4> <keyframe.png> <ref_xiaoya.png>
"""
import os
import sys
import subprocess

import cv2
import numpy as np
from insightface.app import FaceAnalysis

FFMPEG = "~/ffmpeg/ffmpeg.exe"
FFPROBE = "~/ffmpeg/ffprobe.exe"

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))


def probe_dur(mp4):
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", mp4], text=True).strip()
        return float(out)
    except Exception:
        return 5.0


def extract(mp4, t, dst):
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-ss", f"{t:.2f}",
                    "-frames:v", "1", dst], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def imread(p):
    arr = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def emb_of(img):
    """返 [(embedding, det_score, cx] 所有人脸; embedding 已 norm。"""
    fs = app.get(img)
    out = []
    for f in fs:
        e = f.embedding
        n = np.linalg.norm(e)
        out.append((e / n if n > 0 else e, float(f.det_score),
                    float((f.bbox[0] + f.bbox[2]) / 2)))
    return out


def cos(a, b):
    return float(np.dot(a, b))


def pick(img, ref_e):
    """返 (xiaoya_e, tingting_e) 按 vs REF 余弦 max/min。至少 2 脸否则 None。"""
    es = emb_of(img)
    if len(es) < 2:
        return None, None
    scored = sorted([(cos(e, ref_e), e) for (e, _, _) in es], key=lambda x: x[0])
    return scored[-1][1], scored[0][1]   # max=小雅, min=婷婷


def main():
    mp4, kf, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    ref_e = emb_of(imread(ref))[0][0]
    # 关键帧: 取婷婷源脸
    kx, kt = pick(imread(kf), ref_e)
    if kt is None:
        print("[ERR] 关键帧检 <2 脸, 无法定位婷婷源脸")
        return 1
    print(f"关键帧: 小雅vsREF={cos(kx, ref_e):.3f}  婷婷源脸已取\n")

    dur = probe_dur(mp4)
    ts = [round(dur * i / 10, 2) for i in range(1, 10)]   # 9 个采样点
    tmp = mp4 + ".frames"
    os.makedirs(tmp, exist_ok=True)
    print(f"{'t':>5} | {'小雅vsREF':>9} | {'婷婷漂离源':>10} {'婷婷逐帧':>8}")
    print("-" * 48)
    prev_t = None
    xiaoya_min = 1.0
    tt_drift_min, tt_stab_min = 1.0, 1.0
    for t in ts:
        dst = os.path.join(tmp, f"t{t:.2f}.png")
        extract(mp4, t, dst)
        x, tt = pick(imread(dst), ref_e)
        if x is None:
            print(f"{t:>5.1f} |   (检<2脸)")
            continue
        xs = cos(x, ref_e)
        td = cos(tt, kt)
        tseq = cos(tt, prev_t) if prev_t is not None else float("nan")
        print(f"{t:>5.1f} | {xs:>9.3f} | {td:>10.3f} {tseq:>8.3f}")
        xiaoya_min = min(xiaoya_min, xs)
        tt_drift_min = min(tt_drift_min, td)
        if prev_t is not None:
            tt_stab_min = min(tt_stab_min, tseq)
        prev_t = tt
    print("\n=== 判定 ===")
    print(f"小雅身份锁: min={xiaoya_min:.3f}  {'PASS' if xiaoya_min >= 0.45 else 'FAIL'} (≥0.45)")
    print(f"婷婷漂离源: min={tt_drift_min:.3f}  {'PASS' if tt_drift_min >= 0.85 else 'WARN'} (≥0.85)")
    print(f"婷婷自稳定: min={tt_stab_min:.3f}  {'PASS' if tt_stab_min >= 0.85 else 'WARN'} (≥0.85)")
    print("\n[注] 眼睛局部瑕疵本脚本测不到 → 抽末段帧交用户肉眼判")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
