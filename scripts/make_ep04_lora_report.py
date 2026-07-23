"""ep04 LoRA A/B/C 评估报告 (2026-07-23).
   读 gen_ep04_lora_ablation.py 输出的 _ablation_[ABC]/{character}/shot{1..5}.png,
   拼成 5×3 contact sheet + 跑 face_sim 客观测 + 生成 result.json.

用法:
  # 小雅评估
  python scripts/make_ep04_lora_report.py --character xiaoya
  # 林夏评估
  python scripts/make_ep04_lora_report.py --character linxia

输出:
  output/ep04_real/{character}/report/contact_sheet.png  (5 镜 × 3 模式, 240×456/cell)
  output/ep04_real/{character}/report/result.json        (face_sim 矩阵 + 衣服属性自检)
  output/ep04_real/{character}/report/manifest.md        (人读摘要)

[实现说明]
- cell 排版: 列 = A/B/C 模式, 行 = shot1..5. cell 宽 480 (与关键帧原生 w 一致), 高 832.
- contact sheet 大小: 3 列 × 5 行 = 1440×4160; 为简化视觉, cell downscale 25% → 120×208/cell → 360×1040.
- face_sim: 用 fitness venv 子进程调 face_sim.py (per shot, A/B/C 三组都跑, 取 vs ref embedding 的 cosine).
- 衣服属性自检: 对每张 cell, 用 OpenCV HSV 直方图看主色, 写 {top1, top2, top3} 入 result.json.
  xiaoya 期望: top1=white(blouse), top2=black(skirt/hair).
  linxia 期望: top1=misty-blue(top), top2=black(pants), top3=white(shoes/logos).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# manju venv 没有 PIL/cv2 — 该脚本跑 fitness venv (有 insightface/cv2/PIL)
MANJU = "../manju"
FITNESS_VENV = "../fitness-video-pipeline/.venv/Scripts/python.exe"
ABLATION_OUT = os.path.join(MANJU, "output", "ep04_real")

MODES = ("A", "B", "C")
MODE_LABEL = {"A": "no LoRA + face ipa", "B": "LoRA + face ipa", "C": "LoRA + face+body dual ipa"}


def _check_deps():
    import cv2, numpy as np
    return cv2, np


def load_cell(character, mode, shot):
    """返回 path, 或 None (缺图)."""
    p = Path(ABLATION_OUT) / character / f"_ablation_{mode}" / f"shot{shot:02d}.png"
    return p if p.exists() else None


def face_sim_subprocess(ref_face_path, gen_path):
    """跨 venv 调 fitness venv 的 face_sim.py. None=不可测."""
    try:
        r = subprocess.run(
            [FITNESS_VENV, "../manju/scripts/face_sim.py", ref_face_path, gen_path],
            capture_output=True, timeout=60
        )
        out = r.stdout.decode("gbk", errors="ignore")
        # 解析 "cosine_similarity = 0.XXXX" 这一行 (face_sim.py 倒数第 2 行)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("cosine_similarity ="):
                try:
                    return float(line.split("=")[1].strip())
                except (ValueError, IndexError):
                    return None
        # 备用: 输出可能末尾就在 GBK 损坏区, fallback 抓首个浮点数
        for line in out.splitlines():
            line = line.strip()
            if "cosine" in line.lower() and "=" in line:
                try:
                    return float(line.split("=")[1].strip().rstrip(","))  # noqa
                except (ValueError, IndexError):
                    continue
        return None
    except Exception:
        return None


def dominant_colors(cv2, img_path, k=3):
    """返回 [(h, s, v, ratio), ...] 主色 (HSV k-means), ratio 降序."""
    import numpy as np
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # 跳过 black (s<20 & v<20)
    mask = ~((hsv[:, 1] < 20) & (hsv[:, 2] < 20))
    hsv = hsv[mask]
    if len(hsv) < 100:
        return []
    # k-means by bins 8x8x8 quantize
    q = (hsv // 32) * 32
    unique, counts = np.unique(q, axis=0, return_counts=True)
    order = counts.argsort()[::-1][:k]
    out = []
    total = counts.sum()
    for idx in order:
        ratio = counts[idx] / total
        out.append((int(unique[idx][0]), int(unique[idx][1]), int(unique[idx][2]), float(ratio)))
    return out


def hsv_to_name(h, s, v):
    """HSV 名 — 放宽阈值, 区分灰白 / 灰蓝 等。
    - v<35 = 纯黑
    - s<15 = 真灰(几乎无饱和度, 接近空气)
    - 其他: 即使 s 较低(<25)若带 hue, 仍按 hue 报(允许 gray-blue/gray-white)
    - 高 v (>200) + 低 s (<25) = 白
    - 其他看 hue
    """
    if v < 35: return "black"
    # 高亮 + 低饱和 = 白(无论 s 多低, 只要不纯灰就行)
    if v > 210 and s < 30: return "white"
    # s 极低 (空气感灰) → gray
    if s < 12: return "gray"
    # 还在灰区(s 低), 但 hue 显著 → "灰 + hue" 描述, 这里直接归 hue
    # 主色 bin (12 段 ±15°)
    if h < 10 or h >= 170: return "red"
    if h < 22: return "orange"
    if h < 38: return "yellow"
    if h < 62: return "green"
    if h < 92: return "cyan"
    if h < 128: return "blue"
    if h < 158: return "purple"
    return "pink"


def region_top_colors(cv2, np, img_path, region):
    """region = (top, bot, k). 各取 top-k 主色 bin HSV + 占比."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return []
    h_full, w_full = bgr.shape[:2]
    h_start = int(h_full * region[0])
    h_end = int(h_full * region[1])
    crop = bgr[h_start:h_end, :, :]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # 跳过极黑 (S<30 & V<30) — 背景黑色也算, 显式 mask
    mask = ~((hsv[:, 1] < 30) & (hsv[:, 2] < 30))
    hsv = hsv[mask]
    if len(hsv) < 50:
        return []
    q = (hsv // 24) * 24  # bin 12x12x12 更细
    unique, counts = np.unique(q, axis=0, return_counts=True)
    order = counts.argsort()[::-1][:region[2]]
    out = []
    total = counts.sum()
    for idx in order:
        ratio = counts[idx] / total
        out.append((int(unique[idx][0]), int(unique[idx][1]), int(unique[idx][2]), float(ratio)))
    return out


def clothing_check(cv2, np, character, path):
    """按角色期望, 上身 / 裤区 / 鞋区 分别检主色, 写 dict."""
    regions = {
        # region_name: (top %, bot %, k)
        "top": (0.20, 0.50, 3),     # 上衣区
        "pants": (0.45, 0.75, 3),   # 裤区 (略偏下来含高腰)
        "shoes": (0.80, 1.00, 3),   # 鞋/脚
    }
    expected = {
        "xiaoya": {"top": ["white"], "pants": ["black"], "shoes": []},
        "linxia": {"top": ["blue"], "pants": ["black", "gray"], "shoes": ["white"]},
    }[character]
    result = {}
    for rname, region in regions.items():
        cols = region_top_colors(cv2, np, path, region)
        result[rname] = {
            "colors": [
                {"hsv": [h, sat, val], "name": hsv_to_name(h, sat, val), "ratio": round(r, 3)}
                for h, sat, val, r in cols
            ],
            "expected": expected.get(rname, []),
            "matched": [c["name"] for c in [{"hsv": [h, sat, val], "name": hsv_to_name(h, sat, val), "ratio": r} for h, sat, val, r in cols] if c["name"] in expected.get(rname, [])],
        }
    return result


def build_contact_sheet(cv2, np, cells):
    """cells = list of 15 path-or-None, 5 rows × 3 cols. → (1440, 4160, 3) BGR."""
    rows = len(cells)
    cols = len(cells[0])
    cell_w, cell_h = 480, 832  # 原图尺寸
    pad = 8
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad
    # 头部 + 列标签 (mode A/B/C) + 行标签 (shot1..5)
    head_h = 60
    label_w = 120
    sheet = np.full((sheet_h + head_h, sheet_w + label_w, 3), 30, dtype=np.uint8)
    # 列标签
    for j, mode in enumerate(MODES):
        x = label_w + pad + j * (cell_w + pad) + cell_w // 2
        y = head_h // 2
        cv2.putText(sheet, MODE_LABEL[mode], (x - 150, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    # 行标签 + cells
    for i in range(rows):
        y0 = head_h + pad + i * (cell_h + pad)
        cv2.putText(sheet, f"shot{i+1}", (10, y0 + cell_h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2, cv2.LINE_AA)
        for j in range(cols):
            x0 = label_w + pad + j * (cell_w + pad)
            if cells[i][j] and Path(cells[i][j]).exists():
                img = cv2.imread(str(cells[i][j]))
                if img is not None:
                    img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                    sheet[y0:y0 + cell_h, x0:x0 + cell_w] = img
                else:
                    cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h), (0, 0, 80), 2)
                    cv2.putText(sheet, "NO IMG", (x0 + 200, y0 + cell_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 255), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + cell_h), (0, 0, 80), 2)
                cv2.putText(sheet, "MISSING", (x0 + 180, y0 + cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 255), 3, cv2.LINE_AA)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", choices=["xiaoya", "linxia"], required=True)
    ap.add_argument("--out", default=os.path.join(ABLATION_OUT))
    ap.add_argument("--skip-face-sim", action="store_true",
                    help="跳过 face_sim 子进程 (默认跑, 较慢每张 5-15s, 15 张 ≈ 2-4min)")
    args = ap.parse_args()

    cv2, np = _check_deps()
    out_dir = Path(args.out) / args.character / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ref face for face_sim (跟 ablation 脚本的 refs 同步)
    refs = {
        "xiaoya": "../manju/output/xiaoya_real/小雅立绘写实_face.png",
        "linxia": "../manju/output/ep03_lihui/linxia_real_face.png",
    }
    ref_face = refs[args.character]
    if not Path(ref_face).exists():
        print(f"[warn] ref face 不存在: {ref_face}, face_sim 跳过")
        args.skip_face_sim = True

    # 期望衣服 top-N 主色 (HSV 名)
    expected_colors = {
        "xiaoya": ["white", "black"],   # white blouse + black skirt
        "linxia": ["blue", "black", "white"],  # misty-blue + black pants + white shoes
    }

    cells = [[load_cell(args.character, m, s) for m in MODES] for s in range(1, 6)]

    # face_sim + 主色 (per-region)
    result = {
        "character": args.character,
        "ref_face": ref_face if not args.skip_face_sim else None,
        "shots": {},
    }
    for s_idx, s in enumerate(range(1, 6)):
        shot_data = {}
        for m_idx, mode in enumerate(MODES):
            path = cells[s_idx][m_idx]
            entry = {"path": str(path) if path else None}
            if path and Path(path).exists():
                # 区域主色
                entry["clothing"] = clothing_check(cv2, np, args.character, path)
                # 全图辅佐 (legacy 兼容)
                cols = dominant_colors(cv2, path)
                entry["top_colors"] = [
                    {"hsv": [h, sat, val], "name": hsv_to_name(h, sat, val), "ratio": round(r, 3)}
                    for h, sat, val, r in cols
                ]
                if not args.skip_face_sim:
                    fs = face_sim_subprocess(ref_face, str(path))
                    entry["face_sim"] = fs
            shot_data[mode] = entry
        result["shots"][f"shot{s}"] = shot_data

    # contact sheet
    sheet = build_contact_sheet(cv2, np, cells)
    sheet_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"[contact sheet] {sheet_path}  shape={sheet.shape}")

    # 写 result.json + manifest.md
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# ep04 LoRA 评估报告 — {args.character}", ""]
    md.append(f"参考脸: `{ref_face}`")
    md.append(f"contact sheet: `{sheet_path}`")
    md.append("")
    if not args.skip_face_sim:
        md.append("## face_sim 矩阵 (vs ref face)")
        md.append("")
        md.append("| shot | A (baseline) | B (LoRA+face) | C (LoRA+face+body) |")
        md.append("|------|----|----|----|")
        for s in range(1, 6):
            row = [f"shot{s}"]
            for mode in MODES:
                fs = result["shots"][f"shot{s}"][mode].get("face_sim")
                row.append(f"{fs:.3f}" if fs is not None else "-")
            md.append("| " + " | ".join(row) + " |")
        md.append("")
    md.append("## 衣服区域主色 (per-region HSV)")
    md.append("")
    md.append("三区域 top 主色: top(上衣) / pants(裤) / shoes(鞋). 期望色:")
    md.append("- xiaoya: top=white, pants=black")
    md.append("- linxia: top=blue, pants=black或gray, shoes=white")
    md.append("")
    for s in range(1, 6):
        md.append(f"### shot{s}")
        for mode in MODES:
            entry = result["shots"][f"shot{s}"][mode]
            cloth = entry.get("clothing", {})
            if not cloth:
                md.append(f"- **{mode}**: -")
                continue
            md.append(f"- **{mode}**:")
            for rname in ("top", "pants", "shoes"):
                rd = cloth.get(rname, {})
                colors = rd.get("colors", [])
                exp = rd.get("expected", [])
                matched = rd.get("matched", [])
                if not colors:
                    md.append(f"    - {rname}: (无数据, exp={exp})")
                    continue
                top = colors[0]
                top_str = f"{top['name']} ({top['ratio']:.0%})"
                tag = "✅" if (not exp or top['name'] in exp) else "⚠️"
                exp_str = f", exp={exp}" if exp else ""
                md.append(f"    - {rname}: {top_str} {tag}{exp_str}")
        md.append("")
    md_path = out_dir / "manifest.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[report] {json_path}, {md_path}")
    print(f"[next] 看 contact sheet + manifest.md → 用户 eye-gate → 决定 LoRA 留/换")


if __name__ == "__main__":
    sys.exit(main() or 0)
