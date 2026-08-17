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


def build_dual_face_shot_workflow(
    ref_a_name,
    ref_b_name,
    mask_a_name,
    mask_b_name,
    positive,
    negative,
    ckpt=DEFAULT_CKPT,
    width=480,
    height=832,               # Wan i2v 原生桶(与 gen_ep02_real_batch KW/KH 一致)
    seed=None,
    steps=28,
    cfg=6.5,
    sampler="dpmpp_2m",
    scheduler="karras",
    ipadapter_weight=0.9,     # 双锁每张略降(0.95→0.9), 两个强 adapter 叠加易过饱和
    weight_type="linear",
    embeds_scaling="V only",
    denoise=1.0,
    filename_prefix="manju_dual_shot",
):
    """双人锁脸工作流(2026-07-22 Phase 3): 链两个 IPAdapterAdvanced, 各喂一张脸 ref + 一个区域 attn_mask。

    典型: ref_a=小雅脸 + mask_a=左半 → ipa; ref_b=婷婷脸 + mask_b=右半 → ipa2。
    模型链 ckpt → unified → ipa(ref_a, mask_a) → ipa2(ref_b, mask_b) → ksampler。
    attn_mask 走 LoadImage 的 MASK 输出(= alpha 通道), 故 mask PNG 须把区域烘进 alpha
    (区域 alpha=255 白, 区域外 alpha=0)。两张 ref 必须是明显不同人(否则双 adapter 锁混)。
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": ckpt}},
        "ref_a": {"class_type": "LoadImage", "inputs": {"image": ref_a_name}},
        "ref_b": {"class_type": "LoadImage", "inputs": {"image": ref_b_name}},
        "mask_a": {"class_type": "LoadImage", "inputs": {"image": mask_a_name}},
        "mask_b": {"class_type": "LoadImage", "inputs": {"image": mask_b_name}},
        "empty": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
        "unified": {"class_type": "IPAdapterUnifiedLoader",
                    "inputs": {"model": ["ckpt", 0], "preset": "PLUS FACE (portraits)"}},
        # 第一个 adapter: 小雅脸 → 左半区
        "ipa": {"class_type": "IPAdapterAdvanced",
                "inputs": {"model": ["unified", 0],
                           "ipadapter": ["unified", 1],
                           "image": ["ref_a", 0],
                           "weight": ipadapter_weight,
                           "weight_type": weight_type,
                           "combine_embeds": "concat",
                           "start_at": 0.0, "end_at": 1.0,
                           "embeds_scaling": embeds_scaling,
                           "attn_mask": ["mask_a", 1]}},
        # 第二个 adapter: 婷婷脸 → 右半区(吃 ipa 输出模型, 链式)
        "ipa2": {"class_type": "IPAdapterAdvanced",
                 "inputs": {"model": ["ipa", 0],
                            "ipadapter": ["unified", 1],
                            "image": ["ref_b", 0],
                            "weight": ipadapter_weight,
                            "weight_type": weight_type,
                            "combine_embeds": "concat",
                            "start_at": 0.0, "end_at": 1.0,
                            "embeds_scaling": embeds_scaling,
                            "attn_mask": ["mask_b", 1]}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": positive}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": negative}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"model": ["ipa2", 0],
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


def build_face_body_shot_workflow(
    ref_face_name,
    ref_body_name,
    positive,
    negative,
    ckpt=DEFAULT_CKPT,
    width=480,
    height=832,                # Wan i2v 原生桶, 与 build_dual_face_shot_workflow / gen_ep02_real_batch 一致
    seed=None,
    steps=28,
    cfg=6.5,
    sampler="dpmpp_2m",
    scheduler="karras",
    weight_face=0.95,          # 脸 ref weight (沿用 ep02 验证值)
    weight_body=0.70,          # 全身 ref weight (调研推荐值, 待实测调)
    scene_ref_name=None,       # P1-2 场景参考图 (空景) → 锁背景
    weight_scene=0.35,         # 场景 IPAdapter weight (2026-08-17 A/B/C 标定: 0.5 压身份, 0.35 平衡)
    weight_type="linear",
    embeds_scaling="V only",
    denoise=1.0,
    filename_prefix="manju_face_body_shot",
):
    """ep03 双 IPAdapter 锁脸+锁衣服工作流 (2026-07-23 验证)。

    模型链: ckpt → unified → ipa(face ref, weight=0.95) → ipa2(body ref, weight=0.70) → ksampler。
    两个 IPAdapterAdvanced 共用同一 unified loader 的 ipadapter (= 同 model 文件, 方案 B);
    后续可升级方案 A (用 IPAdapterModelLoader 分别加载 face+body adapter model 文件)
    但当前 cubiq/ComfyUI_IPAdapter_plus 同图内 IPAdapterModelLoader + Unified Loader 共存未实测。

    **目的**: 验证"双 IPAdapter 堆叠 (脸 0.95 + 全身 0.70)"能否真锁衣服跨镜一致
    (用户最大痛点 "IPAdapter-Plus FACE 只锁脸不锁身")。

    Args:
        ref_face_name: 脸 ref 文件名 (如 "linxia_real_face.png", 由 crop_face.py 产出)
        ref_body_name: 全身 ref 文件名 (如 "linxia_real.png" 512×912 原图, 不裁)
        weight_face: 脸 IPAdapter weight (默认 0.95, 沿用 ep02)
        weight_body: 全身 IPAdapter weight (默认 0.70, 调研推荐; 可调 0.6-0.8)
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    wf = {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": ckpt}},
        "ref_face": {"class_type": "LoadImage",
                     "inputs": {"image": ref_face_name}},
        "ref_body": {"class_type": "LoadImage",
                     "inputs": {"image": ref_body_name}},
        "empty": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
        "unified": {"class_type": "IPAdapterUnifiedLoader",
                    "inputs": {"model": ["ckpt", 0],
                               "preset": "PLUS FACE (portraits)"}},
        # 第一个 adapter: 脸 ref → 锁脸
        "ipa_face": {"class_type": "IPAdapterAdvanced",
                     "inputs": {"model": ["unified", 0],
                                "ipadapter": ["unified", 1],
                                "image": ["ref_face", 0],
                                "weight": weight_face,
                                "weight_type": weight_type,
                                "combine_embeds": "concat",
                                "start_at": 0.0, "end_at": 1.0,
                                "embeds_scaling": embeds_scaling}},
        # 第二个 adapter: 全身 ref → 尝试锁衣 (链式, 吃 ipa_face 输出)
        "ipa_body": {"class_type": "IPAdapterAdvanced",
                     "inputs": {"model": ["ipa_face", 0],
                                "ipadapter": ["unified", 1],
                                "image": ["ref_body", 0],
                                "weight": weight_body,
                                "weight_type": weight_type,
                                "combine_embeds": "concat",
                                "start_at": 0.0, "end_at": 1.0,
                                "embeds_scaling": embeds_scaling}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": positive}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["ckpt", 1], "text": negative}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"model": ["ipa_body", 0],
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
    # P1-2 场景锁 (可选): 第 3 个 IPAdapter (标准 PLUS) 吃场景参考图, 锁背景.
    # 场景 adapter 用独立 IPAdapterUnifiedLoader (preset "PLUS" 非 FACE) → 标准 ip-adapter-plus_sdxl_vit-h.
    if scene_ref_name:
        wf["ref_scene"] = {"class_type": "LoadImage", "inputs": {"image": scene_ref_name}}
        wf["unified_scene"] = {"class_type": "IPAdapterUnifiedLoader",
                               "inputs": {"model": ["ckpt", 0], "preset": "PLUS (high strength)"}}
        wf["ipa_scene"] = {"class_type": "IPAdapterAdvanced",
                           "inputs": {"model": ["ipa_body", 0],
                                      "ipadapter": ["unified_scene", 1],
                                      "image": ["ref_scene", 0],
                                      "weight": weight_scene,
                                      "weight_type": "linear",
                                      "combine_embeds": "concat",
                                      "start_at": 0.0, "end_at": 1.0,
                                      "embeds_scaling": "V only"}}
        wf["ksampler"]["inputs"]["model"] = ["ipa_scene", 0]
    return wf


def build_facedetailer_workflow(
    ref_image_name,
    frame_image_name,
    positive,
    negative,
    ckpt=DEFAULT_CKPT,
    detector="bbox/face_yolov8m.pt",
    seed=None,
    steps=28,
    cfg=7.0,
    sampler="dpmpp_2m",
    scheduler="karras",
    denoise=0.5,
    ipadapter_weight=0.6,
    weight_type="linear",
    embeds_scaling="V only",
    guide_size=512,
    max_size=1024,
    bbox_threshold=0.5,
    bbox_dilation=10,
    bbox_crop_factor=3.0,
    noise_mask_feather=20,
    force_inpaint=True,
    filename_prefix="manju_fd",
):
    """FaceDetailer 局部脸 inpaint 工作流 (Impact Pack)。

    用途(2026-07-22 治"化妆浓/下眼帘重"): inswapper_128 是 identity-only blend, 保留
    target(i2v 源帧)的妆/眼帘 texture → 换立绘参考没差(已证伪)。改用 FaceDetailer: 检脸
    → 裁脸 crop → img2img 重绘脸区(只动脸, noise_mask 把背景/身体钉死) + IPAdapter(立绘 ref)
    锁身份 → paste 回。prompt 直接指挥"素颜/薄下眼帘", denoise 控改动幅度。

    链: ckpt → unified(PLUS FACE) → ipa(ref 立绘, weight 锁身份) → FaceDetailer(model=ipa)
    ‖ UltralyticsDetectorProvider(face_yolov8m) → bbox_detector
    ‖ LoadImage(frame) → image
    FaceDetailer output[0]=修好的整帧(只脸区变)。
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": ckpt}},
        "ref": {"class_type": "LoadImage",
                "inputs": {"image": ref_image_name}},
        "frame": {"class_type": "LoadImage",
                  "inputs": {"image": frame_image_name}},
        "det": {"class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": detector}},
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
        "fd": {"class_type": "FaceDetailer",
               "inputs": {"image": ["frame", 0],
                          "model": ["ipa", 0],
                          "clip": ["ckpt", 1],
                          "vae": ["ckpt", 2],
                          "positive": ["pos", 0],
                          "negative": ["neg", 0],
                          "bbox_detector": ["det", 0],
                          "wildcard": "",
                          "guide_size": guide_size,
                          "guide_size_for": True,
                          "max_size": max_size,
                          "seed": seed,
                          "steps": steps,
                          "cfg": cfg,
                          "sampler_name": sampler,
                          "scheduler": scheduler,
                          "denoise": denoise,
                          "feather": 5,
                          "noise_mask": True,
                          "force_inpaint": force_inpaint,
                          "bbox_threshold": bbox_threshold,
                          "bbox_dilation": bbox_dilation,
                          "bbox_crop_factor": bbox_crop_factor,
                          "sam_detection_hint": "center-1",
                          "sam_dilation": 0,
                          "sam_threshold": 0.93,
                          "sam_bbox_expansion": 0,
                          "sam_mask_hint_threshold": 0.7,
                          "sam_mask_hint_use_negative": "False",
                          "drop_size": 10,
                          "cycle": 1,
                          "inpaint_model": False,
                          "noise_mask_feather": noise_mask_feather}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["fd", 0], "filename_prefix": filename_prefix}},
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
    """纯文本生图 (CheckpointLoaderSimple + CLIPTextEncode + KSampler).

    默认 SD1.5 DreamShaper (漫画/立绘); P2 角色定妆用 SDXL:
      ckpt="sd_xl_base_1.0.safetensors", width=768, height=1024 (SDXL 单 checkpoint 兼容).
    """
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
            # images = SaveImage; gifs = VHS_VideoCombine (视频); audio = SaveAudio/SaveAudioAdvanced (Music3)
            for key in ("images", "gifs", "audio"):
                for img in node_out.get(key, []) or []:
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

    def generate(self, workflow, out_dir, timeout=900):
        pid = self.queue_prompt(workflow)
        entry = self.wait_for_completion(pid, timeout=timeout)
        return self.fetch_outputs(entry, out_dir)

    def queue_remaining(self):
        try:
            return self._get_json("/queue").get("queue_running", []).__len__() + \
                   len(self._get_json("/queue").get("queue_pending", []))
        except Exception:
            return -1


# ---- LTX-2.3 GGUF i2v (2026-08-06 免费本地 provider) ----

LTX23_MODELS = {
    "unet_gguf": "LTX-2.3-22B-distilled-1.1-Q8_0.gguf",
    "gemma_gguf": "gemma-3-12b-it-IQ4_XS.gguf",
    "text_proj": "ltx-2.3_text_projection_bf16.safetensors",
    "distilled_lora": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "video_vae": "LTX23_video_vae_bf16.safetensors",
}
LTX23_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def build_ltx23_i2v_workflow(
    image_name,
    positive,
    negative,
    width=480,
    height=832,          # 竖屏 (对齐关键帧 480x832)
    length=25,           # 帧数 (step8; 25@24fps≈1.04s 单段)
    seed=None,
    strength=1.0,        # 首帧贴合度
    frame_rate=24,
    filename_prefix="ltx23_i2v",
    models=None,
    sigmas=None,
):
    """LTX-2.3 GGUF 图生视频 (免费本地, 2026-08-06 已验证单段).

    用 LTXVImgToVideoConditionOnly (首帧 latent 注入) 而非旧 LTX-2B 的
    LTXVImgToVideo. 采样器 = distilled 单阶段路径 (CFGGuider cfg=1 + euler_ancestral_cfg_pp
    + ManualSigmas 8 步 flow-match), 对齐 fitness 侧 _ltx23_gguf_768.py.

    ⚠ 本机 12G 显存单段上限 ~25-33 帧 (1.1s), 5s 镜需分段 (连续递进: 每段用上段末帧).
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    m = dict(LTX23_MODELS, **(models or {}))
    return {
        "clip": {"class_type": "DualCLIPLoaderGGUF", "inputs": {
            "clip_name1": m["gemma_gguf"], "clip_name2": m["text_proj"], "type": "ltxv"}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": positive}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": negative}},
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": m["unet_gguf"]}},
        "dist": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0], "lora_name": m["distilled_lora"], "strength_model": 0.5}},
        "loadframe": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "resize": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["loadframe", 0], "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "disabled", "scale_method": "lanczos"}},
        "empty": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width, "height": height, "length": length, "batch_size": 1}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "imgcond": {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {
            "vae": ["vae", 0], "image": ["resize", 0], "latent": ["empty", 0],
            "strength": strength, "bypass": False}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": frame_rate}},
        "guider": {"class_type": "CFGGuider", "inputs": {
            "model": ["dist", 0], "positive": ["cond", 0], "negative": ["cond", 1],
            "cfg": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "samp_sel": {"class_type": "KSamplerSelect",
                     "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas or LTX23_SIGMAS}},
        "ksampler": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["samp_sel", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["imgcond", 0]}},
        "decode": {"class_type": "LTXVTiledVAEDecode", "inputs": {
            "vae": ["vae", 0], "latents": ["ksampler", 0],
            "horizontal_tiles": 1, "vertical_tiles": 1, "overlap": 6, "last_frame_fix": False}},
        "vidcombine": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["decode", 0], "frame_rate": frame_rate, "loop_count": 0,
            "filename_prefix": filename_prefix, "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


# ---- H3 本地 (2026-08-06, 零成本替代 Seedance) ----

H3_MODELS = {
    "fl2va_gguf": "minimax_h3_fl2va_pruned_fp8_Q8_CR.gguf",   # 单人 ImageToVideo
    "ref2va_gguf": "minimax-h3-ref2va-Q8_0.gguf",             # 双人 Ref2VA
    "qwen3vl": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "turbo_lora": "minimax_h3_turbo_4step_ckpt500.safetensors",  # Turbo 加速 LoRA (steps 20→8)
}


def h3_length_from_duration(duration_s: float) -> int:
    """H3 帧数: 17k+5 格 (24fps), 对齐官方公式."""
    return max(5, round(duration_s * 24)) + (5 - (max(5, round(duration_s * 24)) % 17)) % 17


def _h3_base_nodes(positive, width, height, length, seed, filename_prefix,
                   unet_gguf, models=None, turbo=False):
    """H3 公共节点: 模型加载 + 采样链 + 解码.

    turbo=True → 加 MiniMax-H3-Turbo LoRA (steps 20→8) + SigmaShift (video=12,audio=6),
    加速 ~2.5x (治 768x1344 的 2.5h 痛点). 基于 Abiray/MiniMax-H3-Turbo-Lora 官方 workflow.
    """
    m = dict(H3_MODELS, **(models or {}))
    # 模型链: unet → (turbo: lora → sigma_shift) → scheduler/guider
    model_ref = ["unet", 0]
    steps = 20
    if turbo:
        steps = 8
        model_ref = ["lora", 0]
    nodes = {
        "unet": {"class_type": "UnetLoaderGGUFDynamicVRAM", "inputs": {
            "unet_name": unet_gguf}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": m["qwen3vl"], "type": "minimax", "device": "default"}},
        "vae_v": {"class_type": "VAELoader", "inputs": {
            "vae_name": m["video_vae"]}},
        "vae_a": {"class_type": "VAELoader", "inputs": {
            "vae_name": m["audio_vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": positive}},
        "sched": {"class_type": "BasicScheduler", "inputs": {
            "model": model_ref, "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "guider": {"class_type": "BasicGuider", "inputs": {
            "model": model_ref, "conditioning": ["pos", 0]}},
        "samp": {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": "res_multistep"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["samp", 0],
            "sigmas": ["sched", 0], "latent_image": ["cond", 1]}},
        "dec_v": {"class_type": "VAEDecode", "inputs": {
            "samples": ["sampler", 0], "vae": ["vae_v", 0]}},
        "dec_a": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["sampler", 0], "vae": ["vae_a", 0]}},
        "create": {"class_type": "CreateVideo", "inputs": {
            "images": ["dec_v", 0], "audio": ["dec_a", 0], "fps": 24, "loop_count": 0}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["create", 0], "filename_prefix": filename_prefix,
            "format": "mp4", "codec": "auto"}},
    }
    if turbo:
        nodes["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0],
            "lora_name": m["turbo_lora"], "strength_model": 1.0}}
        nodes["sigma_shift"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["lora", 0], "shift_video": 12.0, "shift_audio": 6.0}}
        # 采样模型改为 sigma_shift 输出
        nodes["sched"]["inputs"]["model"] = ["sigma_shift", 0]
        nodes["guider"]["inputs"]["model"] = ["sigma_shift", 0]
    return nodes


def build_h3_i2v_workflow(
    image_name, positive, width=768, height=1344, length=124,
    seed=None, filename_prefix="manju_h3", unet_gguf=None, models=None,
    turbo=False,
):
    """H3 单人 i2v (ImageToVideo + 首帧关键帧). 基于已验证的 _h3_t2v_gguf.json.

    turbo=True → 加 MiniMax-H3-Turbo LoRA (steps 20→8) + SigmaShift, 加速 ~2.5x.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    m = dict(H3_MODELS, **(models or {}))
    wf = _h3_base_nodes(positive, width, height, length, seed, filename_prefix,
                        unet_gguf or m["fl2va_gguf"], models, turbo=turbo)
    # ImageToVideo: first_frame ← LoadImage (首帧关键帧)
    wf["frame"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
        "clip": ["clip", 0], "vae": ["vae_v", 0],
        "first_frame": ["frame", 0], "last_frame": None,
        "prompt": positive, "width": width, "height": height, "length": length}}
    # 采样 latent 从 cond[1] (LATENT 输出)
    wf["sampler"]["inputs"]["latent_image"] = ["cond", 1]
    return wf


def build_h3_ref2va_workflow(
    ref_images, positive, width=768, height=1344, length=124,
    seed=None, filename_prefix="manju_h3_dual", unet_gguf=None, models=None,
    ref_image_size="match", turbo=False, audio=None, audio_slot="ref_audios",
):
    """H3 双人 Ref2VA (多参考图). 基于已验证的 h3_ref2va_local_gguf_fix.json.

    ref_images: list[str] — 参考图文件 (ComfyUI input 名), 每个角色一张.
    positive: "<Picture 1> 角色A 与 <Picture 2> 角色B, ..." 格式.
    turbo=True → 加 MiniMax-H3-Turbo LoRA (steps 20→8) + SigmaShift.
    audio: 台词音频文件 (ComfyUI input 名) → LoadAudio → ref_audios.ref_audio_0.
        H3 用音频音素驱动角色口型 (对话镜). 默认 None (无音频, 环境音).
    audio_slot: "ref_audios" (角色台词驱动口型) | "ref_video_audios" (参考视频音频).
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
    m = dict(H3_MODELS, **(models or {}))
    wf = _h3_base_nodes(positive, width, height, length, seed, filename_prefix,
                        unet_gguf or m["ref2va_gguf"], models, turbo=turbo)
    # N 个 LoadImage + ref_images.ref_image_N 点号 key
    cond_inputs = {
        "clip": ["clip", 0], "vae": ["vae_v", 0], "audio_vae": ["vae_a", 0],
        "prompt": positive, "width": width, "height": height, "length": length,
        "ref_image_size": ref_image_size,
    }
    for i, img in enumerate(ref_images):
        node_id = f"ref{i}"
        wf[node_id] = {"class_type": "LoadImage", "inputs": {"image": img}}
        cond_inputs[f"ref_images.ref_image_{i}"] = [node_id, 0]
    # 台词音频 → LoadAudio → ref_audios/ref_video_audios (驱动口型)
    if audio:
        wf["audio_line"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
        cond_inputs[f"{audio_slot}.ref_audio_0"] = ["audio_line", 0]
    wf["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    wf["sampler"]["inputs"]["latent_image"] = ["cond", 1]
    return wf
