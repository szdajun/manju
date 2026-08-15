#!/usr/bin/env python3
"""scripts/novel_to_script.py — 小说 → 短剧剧本 (P1, 2026-08-15).

小说 TXT → 拆章 → LLM 实体提取 (subject/角色/场景) → 喂 scriptgen → output/epXX/script.json.
orchestrator (`run_ep.py`) 经 load_script 自动消费 (零改动).

用法:
  python scripts/novel_to_script.py --novel novel.txt --ep 20
  python scripts/novel_to_script.py --novel novel.txt --ep 20 --chapter 2 --shots 5 --style real
  python scripts/novel_to_script.py --novel novel.txt --ep 20 --dry-run   # 只打印不调 API
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib import novel  # noqa: E402
from lib import novel_extract  # noqa: E402
from lib.scriptgen import gen_script  # noqa: E402


def _fallback_title(ch_title: str, subject: str) -> str:
    """章节标题太通用 (全文/第N部分) 时用 subject 截取作集标题."""
    if ch_title not in ("全文",) and not ch_title.startswith("第") and ch_title:
        return ch_title
    return subject[:12] if subject else ch_title


def main():
    ap = argparse.ArgumentParser(description="小说 → 短剧剧本: 拆章+提取+生成")
    ap.add_argument("--novel", required=True, help="小说 TXT 路径")
    ap.add_argument("--ep", required=True, help="集数, 如 20")
    ap.add_argument("--chapter", type=int, default=1, help="取第几章 (默认 1)")
    ap.add_argument("--shots", type=int, default=5, help="镜头数 (默认 5)")
    ap.add_argument("--style", choices=["real", "cartoon"], default="real")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印章节/提取/最终 prompt, 不调 API 不写文件")
    ap.add_argument("--out", default=None, help="输出目录 (默认 output/ep{ep})")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=2500)
    args = ap.parse_args()

    # 1. 读 + 拆章
    text = novel.read_novel(args.novel)
    chapters = novel.split_chapters(text)
    if not chapters:
        print(f"[ERR] 章节为空: {args.novel}")
        return 1
    if args.chapter > len(chapters):
        print(f"[ERR] 只有 {len(chapters)} 章, 要第 {args.chapter} 章")
        return 1
    ch = chapters[args.chapter - 1]
    print(f"[novel] {os.path.basename(args.novel)} 共 {len(chapters)} 章; "
          f"取第{args.chapter}章「{ch['title']}」({len(ch['text'])}字)", flush=True)

    # 2. 实体提取
    entities = novel_extract.extract_entities(
        ch["text"], max_tokens=args.max_tokens, temperature=args.temperature)
    print(f"[extract] subject: {entities['subject']}", flush=True)
    print(f"[extract] 角色: {list(entities['characters'].keys())}", flush=True)
    print(f"[extract] 场景: {entities['scenes']}", flush=True)
    if not entities["subject"] or not entities["characters"]:
        print("[ERR] 实体提取不完整 (subject 或 characters 为空)")
        return 1

    # 3. 生成剧本
    title = _fallback_title(ch["title"], entities["subject"])
    result = gen_script(
        subject=entities["subject"], title=title, n_shots=args.shots,
        style=args.style, ep=args.ep, characters_spec=entities["characters"],
        temperature=args.temperature, max_tokens=args.max_tokens,
        dry_run=args.dry_run, out_dir=args.out,
    )
    if args.dry_run:
        print("\n===== 最终剧本 prompt (前 1500 字) =====")
        print(result["prompt"][:1500])
    else:
        out_root = args.out or os.path.join(MANJU, "output", f"ep{args.ep}")
        path = os.path.join(out_root, "script.json")
        print(f"\n已生成: {path}")
        print(f"  {result['title']}  {len(result['shots'])} 镜  角色: "
              f"{list(result['characters'])}")
        print(f"[next] python scripts/run_ep.py --ep {args.ep} --dry-run "
              f"(需先注册 EP_SCRIPTS + 阶段脚本)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
