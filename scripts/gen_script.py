#!/usr/bin/env python3
"""scripts/gen_script.py — Phase 2 剧本 LLM 引擎 CLI (豆包, 2026-08-06).

题材 + 角色设定 → 豆包 LLM → 结构化分镜 JSON → output/epXX/script.json.
orchestrator (`run_ep.py`) 自动探测消费 (零改动).

用法:
  python scripts/gen_script.py --ep 04 --title "装备先到位" \
      --subject "健身女孩第一次去健身房, 立誓自律却先拍照发朋友圈, 幽默反转" --shots 5 --style real
  python scripts/gen_script.py --ep 04 --title x --subject y --dry-run     # 只打印 prompt 不调 API
  python scripts/gen_script.py --ep 04 --title x --subject y \
      --char LINXIA="林夏:健身女孩, 高马尾, 雾蓝运动服" --char "LIWEI=李伟:健身教练, 肌肉型"  # 多角色
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib.scriptgen import gen_script  # noqa: E402

# 默认角色: 复用 ep03 已验证的林夏 (资产/LoRA 延续, 开箱即用)
_DEFAULT_CHARS = {
    "LINXIA": {
        "name": "林夏",
        "brief": "健身女孩, 高马尾, 雾蓝运动服",
        "fingerprint": (
            "1girl, solo, Linxia (林夏), a slim-fit 26-30 year-old Chinese woman, "
            "black hair in a high ponytail, wearing a misty-blue long-sleeve athletic top "
            "with a small white arc logo on the left chest, black high-waist athletic pants "
            "with a thin gray side stripe down the leg, white running shoes, natural makeup, "
            "bright energetic eyes, friendly confident expression, modern athletic wear, gym setting"),
        "face_ref": "output/ep03_lihui/linxia_real_face.png",
        "body_ref": "output/ep03_lihui/linxia_real.png",
        "voice_id": "female-shaonv",
    },
}


def _parse_char(tokens):
    """--char 'LINXIA=林夏:健身女孩, 高马尾' → {LINXIA: {name, brief}}."""
    spec = {}
    for tok in tokens:
        key, _, rest = tok.partition("=")
        name, _, brief = rest.partition(":")
        spec[key.strip()] = {"name": name.strip(), "brief": brief.strip()}
    return spec


def main():
    ap = argparse.ArgumentParser(
        description="剧本 LLM 引擎: 题材+角色 → output/epXX/script.json")
    ap.add_argument("--ep", required=True, help="集数, 如 04")
    ap.add_argument("--title", required=True, help="中文集标题")
    ap.add_argument("--subject", required=True, help="题材/本集内容描述 (中文)")
    ap.add_argument("--shots", type=int, default=5, help="镜头数 (默认 5)")
    ap.add_argument("--style", choices=["real", "cartoon"], default="real",
                    help="画风 (real=写实已验证; cartoon=占位未验证)")
    ap.add_argument("--char", action="append", default=[], metavar="KEY=中文名:人设",
                    help="角色设定, 可多次; 不给则用默认林夏")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印 prompt, 不调 API 不写文件")
    ap.add_argument("--out", default=None,
                    help="输出目录 (默认 output/ep{ep})")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=3000)
    args = ap.parse_args()

    chars = _parse_char(args.char) or _DEFAULT_CHARS
    result = gen_script(
        subject=args.subject, title=args.title, n_shots=args.shots,
        style=args.style, ep=args.ep, characters_spec=chars,
        temperature=args.temperature, max_tokens=args.max_tokens,
        dry_run=args.dry_run, out_dir=args.out,
    )
    if args.dry_run:
        print(result["prompt"])
    else:
        out_root = args.out or os.path.join(MANJU, "output", f"ep{args.ep}")
        path = os.path.join(out_root, "script.json")
        print(f"已生成: {path}")
        print(f"  {result['title']}  {len(result['shots'])} 镜  角色: "
              f"{list(result['characters'])}")
        print(f"[next] python scripts/run_ep.py --ep {args.ep} --dry-run "
              f"(需先注册 EP_SCRIPTS + 阶段脚本)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
