"""本地 MiniMax-Music3 BGM 生成 (2026-08-15, ComfyUI 原生节点, INT8 低显存).

手搭 flow 采样链 (AR 文本编码 → DiT flow → DAV tiled 解码):
  UNETLoader(DiT INT8) → CLIPLoader(type=minimax) → VAELoader(DAV)
  → MiniMaxMusic3TextEncode(caption+lyrics) → EmptyMiniMaxMusic3LatentAudio
  → BasicScheduler + BasicGuider + KSamplerSelect + SamplerCustomAdvanced
  → VAEDecodeAudioTiled → SaveAudio

性能 (12GB 4070 + DynamicVRAM): 20s ~3min (AR 是瓶颈). 短 BGM (20-40s) 可用.
"""
import os

from .comfyui_client import ComfyUIClient

DIT = "minimax_music3_dit_int8_convrot.safetensors"
TEXT_ENC = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
DAV = "minimax_music3_dav.safetensors"

# 健身短剧默认 BGM 描述 (Global Metadata / Vocal Details / Arrangement 三段式)
DEFAULT_BGM_CAPTION = (
    "Global Metadata: Energetic electronic workout track for fitness shorts, "
    "120 BPM, E major, driving four-on-the-floor beat, bright motivational "
    "synth leads, upbeat and punchy throughout, hot pink glam gym atmosphere.\n\n"
    "Vocal Details: No vocals, instrumental.\n\n"
    "Arrangement: Punchy kick on every beat, energetic claps, side-chained "
    "bass, bright arpeggiated synths, building intensity with a clean drop, "
    "clean energetic production.")


def build_music3_workflow(caption, lyrics="", seconds=25.0, seed=42,
                          steps=20, cfg_scale=1.5, top_k=50):
    """Music3 flow 采样链 workflow (API 格式)."""
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": DIT, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": TEXT_ENC, "type": "minimax"}},
        "vae": {"class_type": "VAELoader", "inputs": {
            "vae_name": DAV}},
        "text": {"class_type": "MiniMaxMusic3TextEncode", "inputs": {
            "clip": ["clip", 0], "caption": caption, "lyrics": lyrics,
            "seed": seed, "max_duration": float(seconds),
            "cfg_scale": cfg_scale, "top_k": top_k}},
        "latent": {"class_type": "EmptyMiniMaxMusic3LatentAudio", "inputs": {
            "seconds": float(seconds), "batch_size": 1}},
        "sched": {"class_type": "BasicScheduler", "inputs": {
            "model": ["unet", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "guider": {"class_type": "BasicGuider", "inputs": {
            "model": ["unet", 0], "conditioning": ["text", 0]}},
        "samp": {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": "res_multistep"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["samp", 0],
            "sigmas": ["sched", 0], "latent_image": ["latent", 0]}},
        "dec": {"class_type": "VAEDecodeAudioTiled", "inputs": {
            "samples": ["sampler", 0], "vae": ["vae", 0],
            "tile_size": 512, "overlap": 64}},
        "save": {"class_type": "SaveAudio", "inputs": {
            "audio": ["dec", 0], "filename_prefix": "audio/manju_bgm"}},
    }


def generate_bgm(caption=DEFAULT_BGM_CAPTION, seconds=25.0, seed=42, steps=20,
                 server="http://127.0.0.1:8188", out_dir=None):
    """本地 Music3 生成 BGM → 返回输出文件路径列表 (FLAC).

    out_dir: 默认 ComfyUI/output/audio (ComfyUIClient 拉取产物落点).
    """
    from .comfyui_client import ComfyUIClient as _C
    client = _C(server)
    wf = build_music3_workflow(caption, seconds=seconds, seed=seed, steps=steps)
    if out_dir is None:
        # 默认 ComfyUI/output/audio
        manju = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(os.path.dirname(manju), "ComfyUI", "output", "audio")
    print(f"[music3] 本地生成 {seconds}s BGM (seed={seed}, steps={steps}) ...", flush=True)
    saved = client.generate(wf, out_dir, timeout=3600)
    return saved
