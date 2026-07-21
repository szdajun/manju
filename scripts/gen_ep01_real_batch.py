"""ep01 写实版批量关键帧(10 镜)。SD1.5 DreamShaper + IPAdapter-Plus FACE 锁写实立绘脸。

写实化三换(相对 gen_ep01_batch 漫画版):
  - REF: 漫画立绘 → 主角立绘写实.png(IPAdapter 拉写实风格, 强化照片感)
  - STYLE: japanese manga/lineart/flat → photoreal/50mm/cinematic
  - NEG: 砍掉 "realistic,photorealistic,3d,photograph"(那是反逼漫画的!)+ 加反漫画项
场景描述(SHOTS)风格中性, 复用 gen_shot 已验证分镜; 仅 shot8 去掉 trainer
(IPAdapter 单主体画不出背景壮汉教练, 描述了反而诱导崩)。

逐镜生 → face_sim 抽验 vs 写实立绘(≥0.50 同人)→ 汇总。buffalo_l 整批加载一次。
产物 output/ep01_real/ep01_shotNN_*.png → 用户挑最佳 8 → Seedance i2v(token 紧 10→8)。

用法:
    python scripts/gen_ep01_real_batch.py                 # 全 10 镜
    python scripts/gen_ep01_real_batch.py --shots 1,4,5   # 指定
    python scripts/gen_ep01_real_batch.py --shots 4 --seed 42  # 单镜重抽
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.comfyui_client import ComfyUIClient, build_face_shot_workflow  # noqa: E402

REF = "../manju/output/lihui_real/主角立绘写实.png"
OUT = "../manju/output/ep01_real"

# 角色指纹(写实版, 锁服装/发型/道具; 配合 IPAdapter 锁脸)
# 正脸化后缀(ep01 教训#1: IPAdapter 锁脸需脸清晰正脸, 否则关键帧 face_sim 掉到 0.50 边缘;
#   不写 "neutral expression" 避免压掉 SHOTS 里的笑点表情 — 仅强制脸可见+正脸, 表情由 SHOTS 覆盖)
CHAR = ("1boy, solo, weary exhausted salaryman, 30 years old, deep dark circles under eyes, "
        "messy unkempt short black hair, pale haggard face, light blue plaid shirt, "
        "blue lanyard with id badge around neck, thin frail body, "
        "face clearly visible, looking at viewer, front facing")
STYLE = ("photorealistic, real photograph, 50mm lens, cinematic lighting, shallow depth of field, "
         "detailed skin texture, natural colors, sharp focus, vertical composition, 8k uhd")

# 场景(风格中性, 复用 gen_shot 分镜; shot8 去 trainer)
SHOTS = {
    1: ("close-up, dimly lit office late at night, cold blue glow from computer screen lighting up "
        "his exhausted face, half-empty cup of cold coffee on the desk, dark office"),
    2: ("medium shot, slumped back dead in an office chair, id lanyard hanging down, "
        "completely drained and exhausted, office"),
    3: ("wide shot, standing at the entrance of a bright gym, muscular bodybuilders lifting weights in the background, "
        "protagonist looks small and out of place, still wearing his office id badge, gym entrance"),
    4: ("face turning red straining to lift a small dumbbell with both hands, "
        "background muscular bodybuilder easily curling a large dumbbell one-handed, gym"),
    5: ("medium shot, losing balance on a running treadmill, body tilting backwards about to fall off, "
        "comedic dynamic falling pose, treadmill gym"),
    6: ("close-up, pouring white protein powder into a coffee mug, drinking a sip with whole face "
        "scrunched up in disgust, exaggerated painful grimace"),
    7: ("doing plank pose, whole body shaking violently and trembling, timer on the floor showing 3 seconds, "
        "a fly has landed on the tip of his nose, gym floor"),
    8: ("low angle, collapsed flat on his back on the gym floor with arms and legs spread, "
        "weakly giving a thumbs up from the floor, gym floor"),
    9: ("medium shot, looking at a message notification on his smartphone, standing in the gym, "
        "expression both despairing and relieved, holding phone"),
    10: ("close-up, sitting at an office desk late at night, a small dumbbell placed beside the "
         "keyboard, determined yet helpless tired eyes, blue computer screen glow, office desk"),
}

# 写实 NEG: 砍"realistic/photorealistic/3d/photograph"(漫画版靠它反逼漫风), 加反漫画/反 CG
NEG = ("text, watermark, logo, signature, extra fingers, deformed hands, mutated hands, "
       "bad anatomy, extra limbs, blurry, lowres, jpeg artifacts, "
       "anime, manga, cartoon, illustration, painting, 3d render, cg, lineart, flat shading, "
       "plastic skin, airbrushed, multiple views, reference sheet")


def build_positive(shot, extra=""):
    return f"{CHAR}, {SHOTS[shot]}, {STYLE}" + (f", {extra}" if extra else "")


def _face_sim_pair(ref, shot_path):
    """(sim, n_ref, n_shot)。buffalo_l 缓存函数属性, 整批加载一次。CJK 路径走 fromfile+imdecode。"""
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    if not hasattr(_face_sim_pair, "_app"):
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_sim_pair._app = app
    app = _face_sim_pair._app

    def rd(p):
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

    def primary(p):
        fs = app.get(rd(p))
        if not fs:
            return None, 0
        fs.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        e = fs[0].embedding
        e = e / (np.linalg.norm(e) + 1e-9)
        return e, len(fs)

    ea, na = primary(ref)
    eb, nb = primary(shot_path)
    if ea is None or eb is None:
        return None, na, nb
    return float(np.dot(ea, eb)), na, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--weight", type=float, default=0.95)  # ep01 教训#1: 0.8 脸漂大, 0.95 收紧脸锁
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    assert os.path.exists(REF), f"缺写实立绘: {REF}"
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    c = ComfyUIClient(args.server)
    ref_name = c.upload_image(REF)
    print(f"[upload] ref(写实立绘) -> {ref_name}\n")

    results = []
    for s in shots:
        pos = build_positive(s)
        wf = build_face_shot_workflow(
            ref_name, pos, NEG,
            seed=args.seed, ipadapter_weight=args.weight,
            width=512, height=912, filename_prefix=f"ep01_shot{s:02d}",
        )
        print(f"=== shot {s:2d} ===\n  {SHOTS[s]}", flush=True)
        outs = c.generate(wf, args.out)
        path = outs[0]
        sim, na, nb = _face_sim_pair(REF, path)
        if sim is None:
            flag, simstr = "[no-face]", "N/A"
        elif sim >= 0.5:
            flag, simstr = "[OK]", f"{sim:.3f}"
        else:
            flag, simstr = "[FAIL?]", f"{sim:.3f}"
        print(f"  -> {os.path.basename(path)}  face_sim={simstr}  {flag}  (ref={na} shot={nb})\n", flush=True)
        results.append((s, path, sim, flag))

    print("===== 汇总 =====")
    fails = 0
    for s, p, sim, flag in results:
        simstr = f"{sim:.3f}" if sim is not None else "N/A"
        print(f"  shot {s:2d}  sim={simstr}  {flag}  {os.path.basename(p)}")
        if flag != "[OK]":
            fails += 1
    print(f"\n[batch done] {len(results)} 镜, {fails} 镜待复核(重抽: --shots <号> --seed <新>)")


if __name__ == "__main__":
    main()
