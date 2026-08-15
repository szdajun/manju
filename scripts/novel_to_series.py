#!/usr/bin/env python3
"""scripts/novel_to_series.py — 长篇小说 → 多集剧本 (P1-1, 2026-08-15).

小说 → 拆章 → 分组多集 → 系列角色锁定 (跨集一致) → 每集生成 script.json.

用法:
  python scripts/novel_to_series.py --novel novel.txt --ep-start 21 --episodes 2
  python scripts/novel_to_series.py --novel novel.txt --ep-start 21            # auto 分组
  python scripts/novel_to_series.py --novel novel.txt --ep-start 21 --dry-run  # 只打印不调 API
产物: output/ep{ep_start+i}/script.json (每集, 角色跨集一致)
"""
import argparse
import os
import sys

MANJU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MANJU)

from lib import novel  # noqa: E402
from lib import novel_extract  # noqa: E402
from lib import novel_series  # noqa: E402
from lib.scriptgen import gen_script  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="长篇小说 → 多集剧本 (角色跨集锁定)")
    ap.add_argument("--novel", required=True, help="小说 TXT 路径")
    ap.add_argument("--ep-start", required=True, help="首集号, 如 21")
    ap.add_argument("--episodes", type=int, default=None,
                    help="集数 (默认 auto 按字数分组)")
    ap.add_argument("--shots", type=int, default=5, help="每集镜头数 (默认 5)")
    ap.add_argument("--style", choices=["real", "cartoon"], default="real")
    ap.add_argument("--no-char-lock", action="store_true",
                    help="关闭系列角色锁定 (每集独立提取角色; 默认锁定跨集一致)")
    ap.add_argument("--char-assets", default=None,
                    help="继承上集资产: 指向已生成定妆图的 script.json (如 output/ep20/script.json), "
                         "同名角色复用其 face_ref/body_ref, 免重复生成定妆")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印分组/系列角色/每集 prompt, 不调 API 不写文件")
    ap.add_argument("--out-root", default=None, help="输出根目录 (默认 output)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=2500)
    args = ap.parse_args()

    # 1. 读 + 拆章 + 分组
    text = novel.read_novel(args.novel)
    chapters = novel.split_chapters(text)
    if not chapters:
        print(f"[ERR] 章节为空: {args.novel}")
        return 1
    episodes = novel_series.group_into_episodes(chapters, episodes_n=args.episodes)
    print(f"[novel] {os.path.basename(args.novel)} {len(chapters)} 章 → "
          f"{len(episodes)} 集", flush=True)
    for i, ep in enumerate(episodes):
        print(f"  集{i + 1} [{ep['title']}] {len(ep['chapters'])} 章 "
              f"({sum(len(c['text']) for c in ep['chapters'])}字)", flush=True)

    # 2. 系列角色 (锁定跨集一致)
    if not args.no_char_lock:
        series_chars = novel_series.extract_series_characters(
            chapters, max_tokens=args.max_tokens, temperature=args.temperature)
        print(f"[series-chars] 系列角色: {list(series_chars.keys())}", flush=True)
        if not series_chars:
            print("[WARN] 系列角色提取为空, 退回每集独立提取", flush=True)
            args.no_char_lock = True
    else:
        series_chars = {}

    # 继承上集资产 (face_ref/body_ref): 同名系列角色复用, 免重复生成定妆图
    asset_overrides = {}
    if args.char_assets:
        import json as _json
        src = _json.load(open(args.char_assets, encoding="utf-8"))
        for k, ch in src.get("characters", {}).items():
            if ch.get("face_ref"):
                asset_overrides[k] = {"face_ref": ch["face_ref"], "body_ref": ch.get("body_ref")}
        print(f"[char-assets] 继承 {len(asset_overrides)} 个角色资产", flush=True)

    # 3. 每集生成
    ep_start = int(args.ep_start)
    out_root = args.out_root or os.path.join(MANJU, "output")
    locked_chars = None   # 首集展开的指纹 → 后续集锁定 (角色 DNA 跨集一致)
    for i, ep in enumerate(episodes):
        ep_num = ep_start + i
        ep_text = "\n\n".join(c["text"] for c in ep["chapters"])
        entities = novel_extract.extract_entities(
            ep_text, max_tokens=args.max_tokens, temperature=args.temperature)
        if i == 0:
            chars = series_chars if series_chars else entities["characters"]
        else:
            # 后续集: 用首集展开的 fingerprint 锁定 (防 LLM 每集展开 brief 变体)
            chars = locked_chars or series_chars
        # 继承资产 (face_ref/body_ref): 同名角色复用上集定妆图
        for k in list(chars):
            if k in asset_overrides:
                chars[k].update(asset_overrides[k])
        title = ep["title"] if ep["title"] != "全文" else entities["subject"][:12]
        print(f"\n[ep{ep_num}] subject: {entities['subject']}", flush=True)
        print(f"  chars: {list(chars)}", flush=True)
        if args.dry_run:
            continue
        result = gen_script(
            subject=entities["subject"], title=title, n_shots=args.shots,
            style=args.style, ep=str(ep_num), characters_spec=chars,
            temperature=args.temperature, max_tokens=args.max_tokens,
            out_dir=os.path.join(out_root, f"ep{ep_num}"),
        )
        print(f"  已生成 ep{ep_num}/script.json ({result['title']} "
              f"{len(result['shots'])}镜, 角色 {list(result['characters'])})", flush=True)
        if i == 0:
            # 锁定首集展开的英文指纹 → 后续集复用
            locked_chars = {k: {"name": v.get("name"), "fingerprint": v["fingerprint"]}
                            for k, v in result["characters"].items()}

    if args.dry_run:
        print("\n[dry-run] 未写文件. 去掉 --dry-run 真跑.")
    else:
        print(f"\n[done] {len(episodes)} 集 → {out_root}/ep{ep_start}..{ep_start + len(episodes) - 1}/script.json")
        print(f"[next] 注册 EP_SCRIPTS (orchestrator) + python scripts/run_ep.py --ep {ep_start} --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
