"""tests/test_mux_routing.py — 组装音频路由 (has_voice × has_orig 四象限).

端到端整合 (2026-08-06): mux_seg_audio 自动按 has_voice × has_orig 路由,
provider 无关 (H3 对话/旁白/环境音三模式 + seedance/ltx 无声). 纯函数直接测,
不跑真实 ffmpeg.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import make_ep01 as m  # noqa: E402  gen_seg_i2v/mux_seg_audio/resolve_mux_mode


def test_route_matrix():
    """has_voice × has_orig 四象限 → mix/tts/keep/silent."""
    assert m.resolve_mux_mode(has_voice=True, has_orig=True) == "mix"     # H3 旁白镜
    assert m.resolve_mux_mode(has_voice=True, has_orig=False) == "tts"    # seedance
    assert m.resolve_mux_mode(has_voice=False, has_orig=True) == "keep"   # H3 环境音/对话
    assert m.resolve_mux_mode(has_voice=False, has_orig=False) == "silent"  # 全无音


def test_has_audio_stream_probe():
    """_has_audio_stream: 有音轨 → True, 无 → False (mock ffprobe 输出)."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1\n"
        assert m._has_audio_stream("h3_with_sound.mp4") is True
        mock_run.return_value.stdout = ""
        assert m._has_audio_stream("seedance_silent.mp4") is False


def test_mux_mix_filter_pan_stereo(tmp_path):
    """mix 模式 filter 必须含 pan=stereo (治 TTS 单声道上混 -3dB 衰减) + normalize=0 + alimiter."""
    seg = tmp_path / "h3.mp4"; seg.write_bytes(b"x")
    voice = tmp_path / "v.mp3"; voice.write_bytes(b"x")
    with patch("make_ep01._has_audio_stream", return_value=True) as _has, \
         patch("make_ep01.run") as mock_run:
        m.mux_seg_audio(str(seg), str(voice), str(tmp_path / "out.mp4"))
    fc = mock_run.call_args[0][0]
    fc_str = " ".join(str(x) for x in fc)
    assert "pan=stereo|c0=c0|c1=c0" in fc_str     # 单声道复制不衰减
    assert "normalize=0" in fc_str                # 不压低混音
    assert "alimiter=limit=0.95" in fc_str        # 防削波
    assert "adelay=300," in fc_str or "adelay=300\"" in fc_str  # 单声道单值 delay


def test_gen_seg_i2v_preserves_audio(tmp_path):
    """gen_seg_i2v ffmpeg 命令含 -map 0:a? (保留 H3 原生音, 端到端根因修复)."""
    src = tmp_path / "src.mp4"; src.write_bytes(b"x")
    with patch("make_ep01.run") as mock_run, \
         patch("make_ep01.render_subtitle"):
        m.gen_seg_i2v(str(src), str(tmp_path / "dst.mp4"), None)
    argv = mock_run.call_args[0][0]
    assert "-map" in argv and "0:a?" in argv
    assert "-c:a" in argv and "aac" in argv
