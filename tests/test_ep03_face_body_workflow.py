"""build_face_body_shot_workflow 结构不变量(纯 dict, 不依赖 ComfyUI 运行)。

ep03 关键帧 A/B 实验 builder (2026-07-23):
- 脸 ref 0.95 锁脸 (沿用 ep02 IPAdapter-Plus FACE 范式)
- 全身 ref 0.70 锁衣服 (双 IPAdapterAdvanced 链式)
- 同 cubiq ipadapter model file (方案 B; 后续可升级方案 A 用 IPAdapterModelLoader
  分别加载 face + body adapter model 文件)
- sampler 接第二个 IPAdapter 输出 (= 链式, 两路都生效)
"""
from lib.comfyui_client import build_face_body_shot_workflow


def _by_id(wf):
    return {k: v for k, v in wf.items()}


def _by_class(wf):
    """Return {class_type: [node_ids...]} (允许多个同 class 节点)。"""
    out = {}
    for k, v in wf.items():
        out.setdefault(v["class_type"], []).append(k)
    return out


def test_workflow_returns_dict():
    """基本返回值: dict 格式 (ComfyUI API 格式约定)。"""
    wf = build_face_body_shot_workflow(
        ref_face_name="linxia_real_face.png",
        ref_body_name="linxia_real.png",
        positive="positive prompt",
        negative="negative prompt",
    )
    assert isinstance(wf, dict)
    assert len(wf) > 0


def test_workflow_has_all_required_nodes():
    """必备节点齐全: CheckpointLoaderSimple / 2×LoadImage / IPAdapterUnifiedLoader /
    2×IPAdapterAdvanced / 2×CLIPTextEncode / KSampler / VAEDecode / SaveImage。"""
    wf = build_face_body_shot_workflow(
        "face.png", "body.png", "pos", "neg",
    )
    classes = {n["class_type"] for n in wf.values()}
    required = {
        "CheckpointLoaderSimple", "LoadImage", "IPAdapterUnifiedLoader",
        "IPAdapterAdvanced", "CLIPTextEncode", "KSampler",
        "VAEDecode", "SaveImage", "EmptyLatentImage",
    }
    missing = required - classes
    assert not missing, f"missing node classes: {missing}"
    # 双 LoadImage (face + body)
    load_images = [k for k, v in wf.items() if v["class_type"] == "LoadImage"]
    assert len(load_images) == 2, f"expected 2 LoadImage (face+body), got {len(load_images)}"
    # 双 IPAdapterAdvanced
    adapters = [k for k, v in wf.items() if v["class_type"] == "IPAdapterAdvanced"]
    assert len(adapters) == 2, f"expected 2 IPAdapterAdvanced, got {len(adapters)}"


def test_face_and_body_weights_correct():
    """face weight=0.95, body weight=0.70 (用户拍板)。"""
    wf = build_face_body_shot_workflow(
        "face.png", "body.png", "pos", "neg",
        weight_face=0.95, weight_body=0.70,
    )
    adapters = [v for v in wf.values() if v["class_type"] == "IPAdapterAdvanced"]
    weights = sorted([a["inputs"]["weight"] for a in adapters])
    assert weights == [0.70, 0.95], f"weights should be [0.70, 0.95], got {weights}"


def test_face_and_body_load_distinct_images():
    """face ref 和 body ref 必须分别 LoadImage 节点, 不可共享。"""
    wf = build_face_body_shot_workflow(
        ref_face_name="linxia_face.png",
        ref_body_name="linxia_body.png",
        positive="pos", negative="neg",
    )
    by_class = _by_class(wf)
    load_ids = by_class["LoadImage"]
    loaded = {wf[k]["inputs"]["image"] for k in load_ids}
    assert loaded == {"linxia_face.png", "linxia_body.png"}, (
        f"LoadImage must load face+body refs distinctly, got {loaded}"
    )


def test_face_image_wired_to_face_adapter():
    """face ref (LoadImage #1) → face IPAdapterAdvanced.image。"""
    wf = build_face_body_shot_workflow(
        ref_face_name="linxia_face.png",
        ref_body_name="linxia_body.png",
        positive="pos", negative="neg",
    )
    by_class = _by_class(wf)
    load_ids = by_class["LoadImage"]
    face_load_id = load_ids[0]  # 第 1 个 (insertion order = ref_face)
    face_adapter = next(
        v for v in wf.values()
        if v["class_type"] == "IPAdapterAdvanced" and v["inputs"]["weight"] == 0.95
    )
    assert face_adapter["inputs"]["image"] == [face_load_id, 0], (
        f"face adapter must reference LoadImage[0]={face_load_id}, got {face_adapter['inputs']['image']}"
    )


def test_body_image_wired_to_body_adapter():
    """body ref (LoadImage #2) → body IPAdapterAdvanced.image (weight=0.70)。"""
    wf = build_face_body_shot_workflow(
        ref_face_name="linxia_face.png",
        ref_body_name="linxia_body.png",
        positive="pos", negative="neg",
    )
    by_class = _by_class(wf)
    load_ids = by_class["LoadImage"]
    body_load_id = load_ids[1]  # 第 2 个 LoadImage (body ref)
    body_adapter = next(
        v for v in wf.values()
        if v["class_type"] == "IPAdapterAdvanced" and v["inputs"]["weight"] == 0.70
    )
    assert body_adapter["inputs"]["image"] == [body_load_id, 0], (
        f"body adapter must reference LoadImage[1]={body_load_id}, got {body_adapter['inputs']['image']}"
    )


def test_sampler_wired_to_body_adapter_output_chain():
    """sampler 接 body adapter 的 model 输出 (= 链式 face→body, 两路都生效)。"""
    wf = build_face_body_shot_workflow(
        "face.png", "body.png", "pos", "neg",
    )
    sampler_id = _by_class(wf)["KSampler"][0]
    sampler = wf[sampler_id]
    body_adapter_id = next(
        k for k, v in wf.items()
        if v["class_type"] == "IPAdapterAdvanced" and v["inputs"]["weight"] == 0.70
    )
    # sampler.model[0] 必须 = body adapter node id
    sampler_model_ref = sampler["inputs"]["model"][0]
    assert sampler_model_ref == body_adapter_id, (
        f"sampler must consume body adapter [{body_adapter_id}], got [{sampler_model_ref}]"
    )
    # face adapter 输出必须 = body adapter 输入 (链式 face→body)
    face_adapter_id = next(
        k for k, v in wf.items()
        if v["class_type"] == "IPAdapterAdvanced" and v["inputs"]["weight"] == 0.95
    )
    face_to_body = wf[body_adapter_id]["inputs"]["model"]
    assert face_to_body == [face_adapter_id, 0], (
        f"body adapter must consume face adapter [{face_adapter_id}], got {face_to_body}"
    )


def test_workflow_size_480x832_wan_native():
    """关键帧尺寸 480×832 (Wan i2v 原生桶, 沿用 ep02)。"""
    wf = build_face_body_shot_workflow(
        "face.png", "body.png", "pos", "neg",
        width=480, height=832,
    )
    empty = next(v for v in wf.values() if v["class_type"] == "EmptyLatentImage")
    assert empty["inputs"]["width"] == 480
    assert empty["inputs"]["height"] == 832


def test_unified_loader_preset_plus_face():
    """Unified Loader 用 PLUS FACE preset (沿用 ep02)。"""
    wf = build_face_body_shot_workflow(
        "face.png", "body.png", "pos", "neg",
    )
    unified = next(v for v in wf.values() if v["class_type"] == "IPAdapterUnifiedLoader")
    assert "PLUS FACE" in unified["inputs"]["preset"]


def test_seed_random_by_default():
    """不传 seed → 随机。"""
    wf1 = build_face_body_shot_workflow("f", "b", "p", "n")
    wf2 = build_face_body_shot_workflow("f", "b", "p", "n")
    s1 = next(v for v in wf1.values() if v["class_type"] == "KSampler")["inputs"]["seed"]
    s2 = next(v for v in wf2.values() if v["class_type"] == "KSampler")["inputs"]["seed"]
    # 不强求不等 (理论可能巧合), 但要都是 int
    assert isinstance(s1, int) and isinstance(s2, int)


def test_seed_explicit_propagates():
    """传 seed → sampler 用该值。"""
    wf = build_face_body_shot_workflow("f", "b", "p", "n", seed=42)
    sampler = next(v for v in wf.values() if v["class_type"] == "KSampler")
    assert sampler["inputs"]["seed"] == 42


def test_existing_builders_not_regressed():
    """加新 builder 不能破坏现有 build_face_shot_workflow 和 build_dual_face_shot_workflow。"""
    from lib.comfyui_client import build_face_shot_workflow, build_dual_face_shot_workflow
    wf1 = build_face_shot_workflow("ref.png", "p", "n")
    wf2 = build_dual_face_shot_workflow("a.png", "b.png", "ma.png", "mb.png", "p", "n")
    assert isinstance(wf1, dict) and isinstance(wf2, dict)
    # 单 ref: 只有 1 个 LoadImage + 1 个 IPAdapterAdvanced
    assert sum(1 for v in wf1.values() if v["class_type"] == "LoadImage") == 1
    assert sum(1 for v in wf1.values() if v["class_type"] == "IPAdapterAdvanced") == 1
    # 双 ref: 4 个 LoadImage (2 ref + 2 mask) + 2 个 IPAdapterAdvanced
    assert sum(1 for v in wf2.values() if v["class_type"] == "LoadImage") == 4
    assert sum(1 for v in wf2.values() if v["class_type"] == "IPAdapterAdvanced") == 2