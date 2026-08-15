"""gen_character_assets 定妆图生成单测 (manju venv, mock generate/crop)."""
import os

import pytest

from scripts import gen_character_assets as gca


def test_build_prompts_suffixes():
    fp = "1girl, solo, Linxia, misty-blue top"
    face_p, body_p, neg = gca.build_prompts(fp)
    assert fp in face_p and "head and shoulders" in face_p and "looking at viewer" in face_p
    assert fp in body_p and "full body" in body_p and "facing camera" in body_p
    assert "distorted" in neg and "watermark" in neg


def test_build_prompts_empty_fingerprint():
    face_p, body_p, _ = gca.build_prompts("")
    assert face_p.startswith("portrait") and body_p.startswith("full body")


def test_pending_chars_default_missing_refs():
    script = {"characters": {
        "A": {"face_ref": "a.png", "body_ref": "b.png"},   # 有 ref → 跳过
        "B": {"face_ref": None, "body_ref": None},          # 缺 → 处理
        "C": {"fingerprint": "fp"},                          # 缺 body → 处理
    }}
    assert gca._pending_chars(script, None) == ["B", "C"]


def test_pending_chars_explicit():
    script = {"characters": {"A": {}, "B": {}}}
    assert gca._pending_chars(script, "B") == ["B"]
    assert gca._pending_chars(script, "A,B") == ["A", "B"]


def test_generate_calls_workflow(tmp_path):
    """generate() 用 build_txt2img_workflow + ComfyUIClient, 返回最新落盘文件."""
    # 造一个假 client 记录 generate 调用
    class FakeClient:
        def __init__(self):
            self.calls = []
        def generate(self, wf, out_dir):
            self.calls.append((wf, out_dir))
            out = os.path.join(out_dir, "out1.png")
            open(out, "w").write("x")
            return [out]

    client = FakeClient()
    out_dir = str(tmp_path)
    saved = gca.generate(client, "pos", "neg", "ckpt", 768, 1024, 42, "p", out_dir)
    assert saved  # 返回了落盘文件
    assert len(client.calls) == 1
    wf = client.calls[0][0]
    # 工作流包含 SDXL checkpoint + 尺寸
    assert wf["ckpt"]["inputs"]["ckpt_name"] == "ckpt"
    assert wf["empty"]["inputs"]["width"] == 768
    assert wf["empty"]["inputs"]["height"] == 1024
