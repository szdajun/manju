"""tests/test_video_gen.py — LTX 本地 i2v provider (manju 2026-08-06).

验证 ltx_local_i2v 编排 (分段/连续递进/拼接) + build_ltx23_i2v_workflow 结构.
不调真实 ComfyUI/ffmpeg — mock ComfyUIClient + subprocess.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from lib.video_gen import ltx_local_i2v
from lib.comfyui_client import build_ltx23_i2v_workflow, LTX23_MODELS


# ---- build_ltx23_i2v_workflow 结构 ----

def test_ltx23_workflow_structure():
    """LTX-2.3 i2v 工作流: LTXVImgToVideoConditionOnly + GGUF 模型链."""
    wf = build_ltx23_i2v_workflow("frame.png", "pos", "neg")
    assert wf["unet"]["class_type"] == "UnetLoaderGGUF"
    assert wf["unet"]["inputs"]["unet_name"] == LTX23_MODELS["unet_gguf"]
    assert wf["imgcond"]["class_type"] == "LTXVImgToVideoConditionOnly"
    assert wf["imgcond"]["inputs"]["strength"] == 1.0
    assert wf["imgcond"]["inputs"]["image"] == ["resize", 0]  # 首帧注入
    assert wf["ksampler"]["inputs"]["latent_image"] == ["imgcond", 0]
    assert wf["decode"]["class_type"] == "LTXVTiledVAEDecode"
    # 采样器 = distilled 单阶段 (cfg=1 + euler_ancestral_cfg_pp)
    assert wf["guider"]["inputs"]["cfg"] == 1.0
    assert wf["samp_sel"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"


def test_ltx23_workflow_parameters():
    """尺寸/帧数/seed/前缀可配."""
    wf = build_ltx23_i2v_workflow("f.png", "p", "n", width=512, height=896,
                                  length=65, seed=7, frame_rate=24,
                                  filename_prefix="x")
    assert wf["empty"]["inputs"]["width"] == 512
    assert wf["empty"]["inputs"]["height"] == 896
    assert wf["empty"]["inputs"]["length"] == 65
    assert wf["noise"]["inputs"]["noise_seed"] == 7
    assert wf["vidcombine"]["inputs"]["frame_rate"] == 24
    assert wf["vidcombine"]["inputs"]["filename_prefix"] == "x"


# ---- ltx_local_i2v 编排 (mock) ----

class _FakeClient:
    """fake ComfyUIClient: upload 返回 'f{i}.png', generate 产出假 mp4."""

    def __init__(self, server=None):
        self.uploads = []
        self.gen_calls = []
        self.out_dir = None

    def upload_image(self, path):
        name = os.path.basename(path)
        self.uploads.append(name)
        return f"input/{name}"

    def generate(self, wf, out_dir, timeout=900):
        self.gen_calls.append(wf)
        self.out_dir = out_dir
        self.last_timeout = timeout
        # 产出假 mp4 (模拟 1 段)
        seg = os.path.join(out_dir, f"fake_seg{len(self.gen_calls)}.mp4")
        with open(seg, "wb") as f:
            f.write(b"FAKE")
        return [seg]


def test_ltx_local_i2v_segments(tmp_path):
    """分段 + 连续递进 (每段末帧作下段首帧) + 拼接."""
    image = tmp_path / "kf.png"
    image.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()

    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._concat_videos", return_value=str(out)):
        # duration=2s, segment_seconds=4 → 1 段 (不足 2 段)
        ltx_local_i2v(str(image), "motion", str(out), duration=2,
                      segment_seconds=4.0, segment_frames=97, width=480, height=832)
    assert len(fake.gen_calls) == 1  # 2s ≤ 4s 段 → 1 段
    assert len(fake.uploads) == 1     # 只上传首帧


def test_ltx_local_i2v_multi_segment_continuous(tmp_path):
    """5s 镜 → 2 段 (4s + 1s), 第二段首帧 = 第一段末帧 (连续递进)."""
    image = tmp_path / "kf.png"
    image.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()
    captured = {}

    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._extract_last_frame") as _extract, \
         patch("lib.video_gen._concat_videos") as _concat:
        ltx_local_i2v(str(image), "motion", str(out), duration=5,
                      segment_seconds=4.0, segment_frames=97, width=480, height=832)
        captured["extract"] = _extract
        captured["concat"] = _concat
    assert len(fake.gen_calls) == 2     # 5s = 4s + 1s → 2 段
    assert len(fake.uploads) == 2       # 首帧 + 段1末帧 (连续递进)
    assert captured["extract"].called    # 抽段1末帧
    assert captured["concat"].called     # 拼接
    # 第二段 seed = base + 1 (段间变化)
    assert fake.gen_calls[0]["noise"]["inputs"]["noise_seed"] == 42
    assert fake.gen_calls[1]["noise"]["inputs"]["noise_seed"] == 43


def test_ltx_local_i2v_prompt_flow(tmp_path):
    """prompt 传给每个 LTX 段."""
    image = tmp_path / "kf.png"
    image.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()
    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._concat_videos", return_value=str(out)):
        ltx_local_i2v(str(image), "She rolls shoulders", str(out), duration=2)
    pos_text = fake.gen_calls[0]["pos"]["inputs"]["text"]
    assert "rolls shoulders" in pos_text


# ---- H3 本地 provider (2026-08-06) ----

from lib.comfyui_client import (
    H3_MODELS, build_h3_i2v_workflow, build_h3_ref2va_workflow,
    h3_length_from_duration,
)
from lib.video_gen import h3_local_i2v


def test_h3_length_formula():
    """H3 帧数 17k+5 格 (24fps)."""
    assert h3_length_from_duration(5) == 124
    assert h3_length_from_duration(2) == 56
    assert h3_length_from_duration(1) == 39  # 24帧对齐17格 → 24+15


def test_h3_i2v_workflow_structure():
    """单人 ImageToVideo: 首帧 ← LoadImage, latent ← cond[1], fl2va UNET."""
    wf = build_h3_i2v_workflow("kf.png", "pos", width=768, height=1344,
                               length=124, seed=42)
    assert wf["cond"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert wf["cond"]["inputs"]["first_frame"] == ["frame", 0]
    assert wf["frame"]["class_type"] == "LoadImage"
    assert wf["sampler"]["inputs"]["latent_image"] == ["cond", 1]
    assert wf["unet"]["inputs"]["unet_name"] == H3_MODELS["fl2va_gguf"]
    assert wf["unet"]["class_type"] == "UnetLoaderGGUFDynamicVRAM"
    assert wf["save"]["inputs"]["format"] == "mp4"


def test_h3_ref2va_workflow_structure():
    """双人 Ref2VA: 2 参考图 + ref_images.ref_image_N 点号 key + ref2va UNET."""
    wf = build_h3_ref2va_workflow(["a.png", "b.png"], "<Picture 1> A 与 <Picture 2> B",
                                  width=768, height=1344, length=124, seed=42)
    assert wf["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert wf["ref0"]["class_type"] == "LoadImage"
    assert wf["cond"]["inputs"]["ref_images.ref_image_0"] == ["ref0", 0]
    assert wf["cond"]["inputs"]["ref_images.ref_image_1"] == ["ref1", 0]
    assert wf["unet"]["inputs"]["unet_name"] == H3_MODELS["ref2va_gguf"]
    assert wf["sampler"]["inputs"]["latent_image"] == ["cond", 1]


def test_h3_local_i2v_single(tmp_path):
    """单人: upload 首帧 → build i2v → generate → 产物落 out."""
    image = tmp_path / "kf.png"
    image.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()
    captured = {}

    def _fake_finalize(src, dst):
        with open(dst, "wb") as f:
            f.write(b"FAKE")
        captured["finalized"] = (src, dst)

    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._finalize_h3_output", side_effect=_fake_finalize):
        h3_local_i2v(str(image), "motion", str(out), duration=5,
                     width=768, height=1344, mode="single")
    assert len(fake.gen_calls) == 1
    assert fake.gen_calls[0]["cond"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert fake.gen_calls[0]["cond"]["inputs"]["first_frame"] == ["frame", 0]
    assert os.path.exists(out)


def test_h3_local_i2v_single_dialog_audio(tmp_path):
    """单人对话镜: audio 参数 → 走 Ref2VA 单参考图 + ref_audios 驱动口型."""
    image = tmp_path / "kf.png"; image.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()

    def _fake_finalize(src, dst):
        with open(dst, "wb") as f:
            f.write(b"FAKE")

    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._finalize_h3_output", side_effect=_fake_finalize), \
         patch("lib.video_gen._copy_audio_to_comfy_input",
               return_value="voice_shot01.mp3"):
        h3_local_i2v(str(image), "说话 prompt", str(out), duration=5,
                     mode="single", audio="F:/voices/voice_shot01.mp3")
    assert len(fake.gen_calls) == 1
    wf = fake.gen_calls[0]
    # 对话镜走 Ref2VA (已验证音频驱动口型路径), 单参考图 + 音频
    assert wf["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert "ref_images.ref_image_0" in wf["cond"]["inputs"]
    assert wf["cond"]["inputs"]["ref_audios.ref_audio_0"] == ["audio_line", 0]
    assert wf["audio_line"]["inputs"]["audio"] == "voice_shot01.mp3"
    assert os.path.exists(out)


def test_h3_local_i2v_single_dialog_identity_ref(tmp_path):
    """单人对话镜 identity_ref: refs=[kf, 定妆照] 双参考 (R2V A/B, 2026-08-19)."""
    image = tmp_path / "kf.png"; image.write_bytes(b"PNG")
    ident = tmp_path / "LINXIA_face.png"; ident.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()

    def _fake_finalize(src, dst):
        with open(dst, "wb") as f:
            f.write(b"FAKE")

    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._finalize_h3_output", side_effect=_fake_finalize), \
         patch("lib.video_gen._copy_audio_to_comfy_input",
               return_value="voice_shot01.mp3"):
        h3_local_i2v(str(image), "双参考 prompt", str(out), duration=5,
                     mode="single", audio="F:/voices/voice_shot01.mp3",
                     identity_ref=str(ident))
    assert len(fake.gen_calls) == 1
    wf = fake.gen_calls[0]
    assert wf["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    # 双参考: ref_image_0=kf, ref_image_1=定妆照
    assert "ref_images.ref_image_0" in wf["cond"]["inputs"]
    assert "ref_images.ref_image_1" in wf["cond"]["inputs"]
    assert wf["cond"]["inputs"]["ref_audios.ref_audio_0"] == ["audio_line", 0]
    assert os.path.exists(out)


def test_h3_local_i2v_single_noaudio_image_to_video(tmp_path):
    """单人无 audio (旁白/环境音镜): 仍走 ImageToVideo (不误入 ref2va)."""
    image = tmp_path / "kf.png"; image.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()

    def _fake_finalize(src, dst):
        with open(dst, "wb") as f:
            f.write(b"FAKE")

    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._finalize_h3_output", side_effect=_fake_finalize):
        h3_local_i2v(str(image), "motion", str(out), duration=5, mode="single")
    assert len(fake.gen_calls) == 1
    assert fake.gen_calls[0]["cond"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert "audio_line" not in fake.gen_calls[0]


def test_h3_local_i2v_dual(tmp_path):
    """双人: upload 2 参考图 → build ref2va → 产物落 out."""
    a = tmp_path / "a.png"; a.write_bytes(b"PNG")
    b = tmp_path / "b.png"; b.write_bytes(b"PNG")
    out = tmp_path / "out.mp4"
    fake = _FakeClient()

    def _fake_finalize(src, dst):
        with open(dst, "wb") as f:
            f.write(b"FAKE")

    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake), \
         patch("lib.video_gen._finalize_h3_output", side_effect=_fake_finalize):
        h3_local_i2v(str(a), "<Picture 1> A 与 <Picture 2> B", str(out),
                     duration=5, width=768, height=1344, mode="dual",
                     ref_images=[str(a), str(b)])
    assert len(fake.gen_calls) == 1
    wf = fake.gen_calls[0]
    assert wf["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert "ref_images.ref_image_0" in wf["cond"]["inputs"]
    assert "ref_images.ref_image_1" in wf["cond"]["inputs"]
    assert os.path.exists(out)


def test_h3_local_dual_requires_refs(tmp_path):
    """双人模式缺 ref_images → ValueError."""
    out = tmp_path / "out.mp4"
    fake = _FakeClient()
    with patch("lib.comfyui_client.ComfyUIClient", return_value=fake):
        import pytest as _pt
        with _pt.raises(ValueError):
            h3_local_i2v("kf.png", "p", str(out), mode="dual", ref_images=None)
