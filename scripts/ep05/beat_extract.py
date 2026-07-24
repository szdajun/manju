"""Extract beats from BGM using librosa. Used by Stage 4 to align subtitle + motion cuts."""
import json
import numpy as np
import librosa
from pathlib import Path

def load_bgm(path: str) -> tuple[np.ndarray, int]:
    """Load mp3/wav into mono float32 waveform + sample rate."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def extract_beats(y: np.ndarray, sr: int) -> dict:
    """Run librosa beat tracker. Returns bpm + beat_times in seconds."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    return {"bpm": float(tempo), "beat_times": beat_times}

def cuts_from_beats(beat_times: list[float], n_cuts: int = 4) -> list[float]:
    """Pick the first n_cuts beat times as scene cut anchors."""
    return beat_times[:n_cuts]

def save_beats_json(beats: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(beats, f, indent=2, ensure_ascii=False)
