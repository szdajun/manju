import json
import os
import numpy as np
import pytest
from scripts.ep05.beat_extract import load_bgm, extract_beats, cuts_from_beats

def test_load_bgm_returns_y_and_sr():
    # 用一个内置 sine-wave wav 试
    import scipy.io.wavfile as wav
    sr = 22050
    t = np.linspace(0, 1, sr)
    y = np.sin(2 * np.pi * 440 * t)
    tmp = "tests/_tmp_loadbgm.wav"
    wav.write(tmp, sr, (y * 32767).astype(np.int16))
    y_out, sr_out = load_bgm(tmp)
    os.remove(tmp)
    assert sr_out == sr
    assert len(y_out) == len(y)

def test_extract_beats_returns_dict():
    sr = 22050
    t = np.linspace(0, 10, 10 * sr)
    # 124 BPM 节奏: 每 60/124 ≈ 0.4839s 一个 beat 标记
    y = np.zeros_like(t)
    for i in range(int(10 / (60/124))):
        idx = int(i * (60/124) * sr)
        if idx < len(y):
            y[idx:idx+100] = 1.0
    out = extract_beats(y, sr)
    assert "bpm" in out
    assert "beat_times" in out
    assert len(out["beat_times"]) >= 5
