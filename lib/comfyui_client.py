"""ComfyUI HTTP API client + IPAdapter-Plus 角色一致性生图工作流。

零第三方依赖(纯 urllib)。manju 通过 HTTP(localhost:8188)调本地 ComfyUI,
不碰 ComfyUI venv,子进程/HTTP 解耦 —— manju 物理隔离原则。

工作流: SD1.5 checkpoint → IPAdapterUnifiedLoader(PLUS FACE 自动配 ViT-H + plus-face)
→ IPAdapterAdvanced(立绘当脸参考) → CLIPTextEncode → KSampler → VAE → Save。
"""
import json
import os
import time
import urllib.parse
import urllib.request
import uuid

DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_CKPT = "DreamShaper_8_pruned.safetensors"


# ---------------- 工作流构造 ----------------
def build_face_shot_workflow(
    ref_image_name,
    positive,
    negative,
    ckpt=DEFAULT_CKPT,
    width=512,
    height=912,                # 512x912 ≈ 9:16 竖屏, SD1.5 友好(8 的倍数)
    seed=None,
    steps=28,
    cfg=6.5,
    sampler="dpmpp_2m",
    scheduler="karras",
    ipadapter_weight=0.8,
    weight_type="linear",
    embeds_scaling="V only",
    denoise=1.0,
    filename_prefix="manju_shot",
):
    """构造 ComfyUI API 格式工作流 dict(用立绘 ref 锁脸生镜)."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": ckpt}},
        "ref": {"class_type": "LoadImage",
                "inputs": {"image": ref_image_name}},
        "empty": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
        "unified": {"class_type": "IPAdapterUnifiedLoader",
                    "inputs": {"model": ["ckpt", 0],
                               "preset": "PLUS FACE (portraits)"}},
        "ipa": {"class_type": "IPAdapterAdvanced",
                "inputs": {"model": ["unified", 0],
                           "ipadapter": ["unified", 1],
                           "image": ["ref", 0],
                           "weight": ipadapter_weight,
                           "weight_type": weight_type,
                           "combine_embeds": "concat",
                           "start_at": 0.0,
                           "end_at": 1.0,
                           "embeds_scaling": embeds_scaling}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": positive}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": negative}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"model": ["ipa", 0],
                                "positive": ["pos", 0],
                                "negative": ["neg", 0],
                                "latent_image": ["empty", 0],
                                "seed": seed,
                                "steps": steps,
                                "cfg": cfg,
                                "sampler_name": sampler,
                                "scheduler": scheduler,
                                "denoise": denoise}},
        "vae": {"class_type": "VAEDecode",
                "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["vae", 0], "filename_prefix": filename_prefix}},
    }


def build_txt2img_workflow(
    positive,
    negative,
    ckpt=DEFAULT_CKPT,
    width=512,
    height=912,            # 9:16 竖版, 匹配 manju + Seedance i2v
    seed=None,
    steps=32,
    cfg=6.5,
    sampler="dpmpp_2m",
    scheduler="karras",
    filename_prefix="manju_lihui",
):
    """SD1.5 纯文本生图(写实立绘用; 立绘是脸源, 不需 IPAdapter 锁脸)。"""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": ckpt}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": positive}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": negative}},
        "empty": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"model": ["ckpt", 0],
                                "positive": ["pos", 0],
                                "negative": ["neg", 0],
                                "latent_image": ["empty", 0],
                                "seed": seed,
                                "steps": steps,
                                "cfg": cfg,
                                "sampler_name": sampler,
                                "scheduler": scheduler,
                                "denoise": 1.0}},
        "vae": {"class_type": "VAEDecode",
                "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["vae", 0], "filename_prefix": filename_prefix}},
    }


def build_ltx_i2v_workflow(
    image_name,
    positive,
    negative,
    ckpt="ltx-video-2b-v0.9.5.safetensors",
    width=512,
    height=768,            # 竖屏 9:16-ish(LTX 对 32 倍数友好)
    length=41,             # 帧数 step8; 41@16fps≈2.5s / @8fps≈5s
    seed=None,
    steps=30,
    cfg=3.0,               # LTX rectified-flow 用低 cfg
    sampler="euler",
    scheduler="sgm_uniform",   # LTX 推荐调度
    strength=1.0,          # 首帧贴合度
    frame_rate=16,
    filename_prefix="manju_i2v",
):
    """LTX-Video 图生视频工作流(吃静帧出 ~2-5s 短动画)。

    漫剧静帧本身已 IPAdapter 锁脸, i2v 继承角色一致性; prompt 描述"要发生的运动"。
    LTXVImgToVideo 把图 encode 进 latent 做首帧锚定 + 处理 conditioning,
    标准 KSampler(euler / sgm_uniform / denoise=1.0) 生成整段 → VAEDecode → VHS 存 mp4。
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": ckpt}},
        "clip_loader": {"class_type": "CLIPLoader", "inputs": {
                "clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "ltxv"}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip_loader", 0], "text": positive}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip_loader", 0], "text": negative}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "ms": {"class_type": "ModelSamplingLTXV", "inputs": {
                "model": ["ckpt", 0], "max_shift": 2.05, "base_shift": 0.95}},
        "i2v": {"class_type": "LTXVImgToVideo", "inputs": {
                "positive": ["pos", 0], "negative": ["neg", 0],
                "vae": ["ckpt", 2], "image": ["img", 0],
                "width": width, "height": height, "length": length,
                "batch_size": 1, "strength": strength}},
        "ksampler": {"class_type": "KSampler", "inputs": {
                "model": ["ms", 0],
                "positive": ["i2v", 0], "negative": ["i2v", 1],
                "latent_image": ["i2v", 2],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler,
                "denoise": 1.0}},
        "vd": {"class_type": "VAEDecode",
               "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "VHS_VideoCombine", "inputs": {
                "images": ["vd", 0], "frame_rate": frame_rate,
                "loop_count": 0, "filename_prefix": filename_prefix,
                "format": "video/h264-mp4", "pingpong": False,
                "save_output": True}},
    }


def build_wan_i2v_workflow(
    image_name,
    positive,
    negative,
    width=512,
    height=768,            # 竖屏; 须 8 的倍数(Wan VAE 空间下采样 ×8)
    length=81,             # 帧数; 81=4k+1, 16fps≈5s
    steps=30,
    cfg=6.0,
    shift=5.0,             # Wan flow-match shift
    seed=None,
    scheduler="unipc",     # 通用(避开实验性 flowmatch_pusa)
    blocks_to_swap=20,     # 5B 共 30 块; 12GB swap 20 块保险
    attention_mode="sdpa", # sageattn 更快(需包); sdpa torch 原生稳
    frame_rate=16,
    model="Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors",
    t5_model="umt5-xxl-enc-fp8_e4m3fn.safetensors",
    t5_quant="fp8_e4m3fn",   # fp8 文件配 fp8 量化(省一半 T5 显存); disabled 会 dequant 出 9.5GB bf16
    t5_force_offload=True,   # 编码后把 T5 挪回 CPU(= kijai 示例默认), 给 5B 采样腾 ~5GB. torch 2.13.0+cu126 的 module.to(cpu) 段错误曾逼得临时关掉它; 已降级 torch→2.7.1+cu126 修复(2026-07-20), 恢复 canonical 配置.
    t5_device="cpu",         # 12GB 显存(RTX 4070)铁律: T5 上 GPU 编码会 OOM. precision 仅 [fp32,bf16] 无 fp8 compute → fp8 文件(4.75GB)上 GPU 被转 bf16(~9.5GB) + VAE + CUDA context > 12GB(2026-07-20 smoke 实测 utils.py:191 value.to(device) OOM). CPU 编码 umt5 ~20s/prompt, 10 镜≈3-4min 可接受. 大显存机器可改 "gpu".
    t5_disk_cache=True,      # CPU T5 编码慢(~10s/layer, 24 层≈4min/prompt)→ 开磁盘缓存(custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache). 同 prompt 二次起命中缓存秒回; smoke 固定 prompt 只付一次成本, 量产 10 镜各 prompt 也只编码一次.
    vae_model="Wan2_2_VAE_bf16.safetensors",
    vae_tile=272,
    filename_prefix="manju_wan",
):
    """Wan2.2-TI2V-5B 图生视频工作流(ComfyUI API 格式)。

    单 5B 模型(2026-07-20 拍板, 替原 14B/1.3B 分工)。节点链以 ComfyUI-WanVideoWrapper
    /object_info + example_workflows/wanvideo_2_2_5B_I2V_example_WIP.json 实测为准:
      LoadImage → WanVideoEncode(首帧→latent) → WanVideoEmptyEmbeds(w/h/num_frames)
      ‖ LoadWanVideoT5TextEncoder + WanVideoTextEncode
      ‖ WanVideoModelLoader(+WanVideoBlockSwap) → WanVideoSampler → WanVideoDecode(tiled)
      → VHS_VideoCombine。
    图像条件走 VAE encode(5B-TI2V 无需 clip_vision)。12GB: offload_device + sdpa + block_swap。
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    return {
        "bswap": {"class_type": "WanVideoBlockSwap", "inputs": {
            "blocks_to_swap": blocks_to_swap, "offload_img_emb": False, "offload_txt_emb": False}},
        "model": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": model, "base_precision": "bf16", "quantization": "fp8_e4m3fn_scaled",
            "load_device": "offload_device", "attention_mode": attention_mode,
            "block_swap_args": ["bswap", 0]}},
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": vae_model, "precision": "bf16"}},
        "t5": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {
            "model_name": t5_model, "precision": "bf16",
            "load_device": "offload_device", "quantization": t5_quant}},
        "text": {"class_type": "WanVideoTextEncode", "inputs": {
            "positive_prompt": positive, "negative_prompt": negative,
            "t5": ["t5", 0], "force_offload": t5_force_offload, "device": t5_device,
            "use_disk_cache": t5_disk_cache}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "enc": {"class_type": "WanVideoEncode", "inputs": {
            "vae": ["vae", 0], "image": ["img", 0], "enable_vae_tiling": True,
            "tile_x": vae_tile, "tile_y": vae_tile,
            "tile_stride_x": vae_tile // 2, "tile_stride_y": vae_tile // 2}},
        "embeds": {"class_type": "WanVideoEmptyEmbeds", "inputs": {
            "width": width, "height": height, "num_frames": length,
            "extra_latents": ["enc", 0]}},
        "sampler": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["model", 0], "image_embeds": ["embeds", 0],
            "text_embeds": ["text", 0],
            "steps": steps, "cfg": cfg, "shift": shift, "seed": seed,
            "force_offload": True, "scheduler": scheduler, "riflex_freq_index": 0}},
        "dec": {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["vae", 0], "samples": ["sampler", 0], "enable_vae_tiling": True,
            "tile_x": vae_tile, "tile_y": vae_tile,
            "tile_stride_x": vae_tile // 2, "tile_stride_y": vae_tile // 2}},
        "save": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["dec", 0], "frame_rate": frame_rate, "loop_count": 0,
            "filename_prefix": filename_prefix, "format": "video/h264-mp4",
            "pix_fmt": "yuv420p", "crf": 19, "pingpong": False, "save_output": True}},
    }


# ---------------- HTTP client ----------------
class ComfyUIClient:
    def __init__(self, server=DEFAULT_SERVER):
        self.server = server.rstrip("/")

    def _post_json(self, path, payload, timeout=60):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.server + path, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def _get_json(self, path, timeout=60):
        with urllib.request.urlopen(self.server + path, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def upload_image(self, local_path, image_type="input", overwrite=True):
        """上传本地图片到 ComfyUI/input/, 返回服务器侧文件名."""
        name = os.path.basename(local_path)
        boundary = "----manju" + uuid.uuid4().hex
        with open(local_path, "rb") as f:
            filedata = f.read()
        crlf = b"\r\n"
        parts = [
            (f"--{boundary}{crlf.decode()}"
             f'Content-Disposition: form-data; name="image"; filename="{name}"{crlf.decode()}'
             f"Content-Type: application/octet-stream{crlf.decode()}{crlf.decode()}").encode(),
            filedata, crlf,
            (f"--{boundary}{crlf.decode()}"
             f'Content-Disposition: form-data; name="type"{crlf.decode()}{crlf.decode()}'
             f"{image_type}{crlf.decode()}").encode(),
            (f"--{boundary}{crlf.decode()}"
             f'Content-Disposition: form-data; name="overwrite"{crlf.decode()}{crlf.decode()}'
             f"{str(overwrite).lower()}{crlf.decode()}").encode(),
            f"--{boundary}--{crlf.decode()}".encode(),
        ]
        body = b"".join(parts)
        req = urllib.request.Request(
            self.server + "/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode())
        return resp.get("name", name)

    def queue_prompt(self, workflow):
        return self._post_json("/prompt", {"prompt": workflow})["prompt_id"]

    def get_history(self, prompt_id):
        return self._get_json(f"/history/{prompt_id}").get(prompt_id)

    def wait_for_completion(self, prompt_id, timeout=900, poll=2.0):
        start = time.time()
        last = None
        while time.time() - start < timeout:
            entry = self.get_history(prompt_id)
            if entry is not None:
                status = entry.get("status", {}) or {}
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI prompt error: {status}")
                return entry
            time.sleep(poll)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout}s (last={last})")

    def fetch_outputs(self, history_entry, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for _node_id, node_out in (history_entry.get("outputs") or {}).items():
            for img in node_out.get("images", []):
                fname = img["filename"]
                q = urllib.parse.urlencode({
                    "filename": fname,
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
                with urllib.request.urlopen(f"{self.server}/view?{q}", timeout=120) as r:
                    data = r.read()
                local = os.path.join(out_dir, fname)
                with open(local, "wb") as f:
                    f.write(data)
                saved.append(local)
        return saved

    def generate(self, workflow, out_dir):
        pid = self.queue_prompt(workflow)
        entry = self.wait_for_completion(pid)
        return self.fetch_outputs(entry, out_dir)

    def queue_remaining(self):
        try:
            return self._get_json("/queue").get("queue_running", []).__len__() + \
                   len(self._get_json("/queue").get("queue_pending", []))
        except Exception:
            return -1
