"""build_wan_i2v_workflow 结构不变量(纯 dict, 不依赖 ComfyUI 运行, manju venv 可跑)。

节点链以 ComfyUI-WanVideoWrapper /object_info 实测为准(非 KSampler)。
"""
from lib.comfyui_client import build_wan_i2v_workflow

REQUIRED_CLASSES = {
    "WanVideoModelLoader", "WanVideoBlockSwap", "WanVideoVAELoader",
    "LoadWanVideoT5TextEncoder", "WanVideoTextEncode", "LoadImage",
    "WanVideoEncode", "WanVideoEmptyEmbeds", "WanVideoSampler",
    "WanVideoDecode", "VHS_VideoCombine",
}


def _by_class(wf):
    return {v["class_type"]: k for k, v in wf.items()}


def test_wan_workflow_has_all_required_nodes():
    wf = build_wan_i2v_workflow("img.png", "a man walks", "blurry", length=81, steps=30)
    classes = {n["class_type"] for n in wf.values()}
    missing = REQUIRED_CLASSES - classes
    assert not missing, f"missing nodes: {missing}"


def test_wan_workflow_empty_embeds_dims_and_frames():
    wf = build_wan_i2v_workflow("img.png", "p", "n", width=512, height=768, length=81)
    emb = [n for n in wf.values() if n["class_type"] == "WanVideoEmptyEmbeds"][0]
    assert emb["inputs"]["width"] == 512
    assert emb["inputs"]["height"] == 768
    assert emb["inputs"]["num_frames"] == 81


def test_wan_workflow_5b_model_and_fp8_quant():
    wf = build_wan_i2v_workflow("i", "p", "n")
    loader = [n for n in wf.values() if n["class_type"] == "WanVideoModelLoader"][0]
    assert "5B" in loader["inputs"]["model"]
    assert loader["inputs"]["quantization"] == "fp8_e4m3fn_scaled"
    assert loader["inputs"]["load_device"] == "offload_device"


def test_wan_workflow_blockswap_wired():
    wf = build_wan_i2v_workflow("i", "p", "n", blocks_to_swap=20)
    bc = _by_class(wf)
    loader = wf[bc["WanVideoModelLoader"]]
    assert loader["inputs"]["block_swap_args"] == [bc["WanVideoBlockSwap"], 0]
    assert wf[bc["WanVideoBlockSwap"]]["inputs"]["blocks_to_swap"] == 20


def test_wan_workflow_sampler_wiring_and_seed():
    wf = build_wan_i2v_workflow("i", "p", "n", seed=42, steps=25, cfg=5.0)
    bc = _by_class(wf)
    s = wf[bc["WanVideoSampler"]]["inputs"]
    assert s["model"] == [bc["WanVideoModelLoader"], 0]
    assert s["image_embeds"] == [bc["WanVideoEmptyEmbeds"], 0]
    assert s["text_embeds"] == [bc["WanVideoTextEncode"], 0]
    assert s["seed"] == 42
    assert s["steps"] == 25
    assert s["cfg"] == 5.0


def test_wan_workflow_image_cond_via_vae_encode():
    # 5B-TI2V 图像条件 = WanVideoEncode 首帧 latent → EmptyEmbeds.extra_latents(无 clip_vision)
    wf = build_wan_i2v_workflow("i", "p", "n")
    bc = _by_class(wf)
    enc = wf[bc["WanVideoEncode"]]["inputs"]
    assert enc["image"] == [bc["LoadImage"], 0]
    assert enc["vae"] == [bc["WanVideoVAELoader"], 0]
    assert wf[bc["WanVideoEmptyEmbeds"]]["inputs"]["extra_latents"] == [bc["WanVideoEncode"], 0]
    assert "CLIPVision" not in "".join(wf.keys())


def test_wan_workflow_seed_default_random():
    def seed_of(wf):
        return next(n["inputs"]["seed"] for n in wf.values()
                    if n["class_type"] == "WanVideoSampler")
    assert seed_of(build_wan_i2v_workflow("i", "p", "n")) != \
        seed_of(build_wan_i2v_workflow("i", "p", "n"))


def test_wan_workflow_t5_fp8_quant():
    # 2026-07-20 smoke 段错误修复守门: fp8 T5 文件必须配 fp8_e4m3fn 量化.
    # quantization=disabled 会 dequant 出 bf16 张量, 在 force_offload 的 .to() 里段错误
    # (ComfyUI 进程 access violation, b96apkhoy exit 139).
    wf = build_wan_i2v_workflow("i", "p", "n")
    bc = _by_class(wf)
    t5 = wf[bc["LoadWanVideoT5TextEncoder"]]["inputs"]
    assert t5["quantization"] == "fp8_e4m3fn", \
        "T5 量化必须 fp8_e4m3fn(disabled 触发段错误)"


def test_wan_workflow_t5_force_offload_enabled():
    # canonical 配置守门: 编码后 offload T5 给 5B 采样腾 VRAM(kijai 示例默认).
    # 历史坑: torch 2.13.0+cu126 的 module.to(cpu) 段错误曾逼得临时关掉它,
    # 已降级 torch→2.7.1+cu126 修复(2026-07-20). 不得把 ComfyUI pyenv 的 torch 升回 2.13.
    wf = build_wan_i2v_workflow("i", "p", "n")
    bc = _by_class(wf)
    text = wf[bc["WanVideoTextEncode"]]["inputs"]
    assert text["force_offload"] is True, \
        "T5 force_offload 应开(给 5B 腾 VRAM); ComfyUI pyenv torch 必须 2.7.x 不能 2.13"


def test_wan_workflow_t5_device_cpu():
    # 12GB 显存铁律守门: T5 必须在 CPU 编码(device='cpu').
    # precision 仅 [fp32,bf16] 无 fp8 compute, fp8 T5 文件(4.75GB)上 GPU 转 bf16(~9.5GB)
    # + VAE + CUDA context > 12GB → torch.OutOfMemoryError (2026-07-20 smoke utils.py:191).
    wf = build_wan_i2v_workflow("i", "p", "n")
    bc = _by_class(wf)
    text = wf[bc["WanVideoTextEncode"]]["inputs"]
    assert text["device"] == "cpu", \
        "T5 device 必须 'cpu'(12GB OOM); 大显存机器才可改 'gpu'"
