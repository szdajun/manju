"""ep04 LoRA 加载 graph 单元测试 (2026-07-23)。

测: build_face_shot_workflow + Load LoRA 节点注入后, graph 仍合法 (Load LoRA 节点在 ckpt 和 unified 之间)。
"""
from lib.comfyui_client import build_face_shot_workflow, build_face_body_shot_workflow


def _inject_lora(wf, lora_name="xiaoya_lora.safetensors", strength=0.85):
    """手动注入 Load LoRA 节点, 串在 ckpt 与 unified 之间."""
    lora_node = {
        "class_type": "LoraLoader",
        "inputs": {"model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": lora_name,
                   "strength_model": strength, "strength_clip": strength},
    }
    wf["lora"] = lora_node
    wf["unified"]["inputs"]["model"] = ["lora", 0]
    wf["pos"]["inputs"]["clip"] = ["lora", 1]
    wf["neg"]["inputs"]["clip"] = ["lora", 1]
    return wf


def test_lora_loaded_into_face_workflow():
    wf = build_face_shot_workflow("face.png", "p", "n", width=480, height=832, seed=42, ipadapter_weight=0.95)
    wf = _inject_lora(wf)
    # 必备节点: LoraLoader
    classes = {v["class_type"] for v in wf.values()}
    assert "LoraLoader" in classes, f"LoraLoader missing: classes={classes}"
    # unified.model 必须指向 lora 节点输出
    lora_id = next(k for k, v in wf.items() if v["class_type"] == "LoraLoader")
    assert wf["unified"]["inputs"]["model"] == [lora_id, 0]
    # pos/neg clip 都从 lora 来
    assert wf["pos"]["inputs"]["clip"] == [lora_id, 1]
    assert wf["neg"]["inputs"]["clip"] == [lora_id, 1]
    # 单 IPAdapter 模式下应只有 1 个 IPAdapterAdvanced, weight 0.95
    ipas = [v for v in wf.values() if v["class_type"] == "IPAdapterAdvanced"]
    assert len(ipas) == 1
    assert ipas[0]["inputs"]["weight"] == 0.95


def test_lora_strength_default():
    """strength 默认 0.85 (ep02 实测 0.95 face 过强, 0.85 平衡)."""
    wf = build_face_shot_workflow("face.png", "p", "n")
    wf = _inject_lora(wf, strength=0.85)
    lora = next(v for v in wf.values() if v["class_type"] == "LoraLoader")
    assert lora["inputs"]["strength_model"] == 0.85
    assert lora["inputs"]["strength_clip"] == 0.85


def test_existing_builders_not_regressed():
    """ep04 测试不能破坏 ep02/ep03 builder."""
    from lib.comfyui_client import build_face_shot_workflow, build_dual_face_shot_workflow, build_face_body_shot_workflow
    assert isinstance(build_face_shot_workflow("r", "p", "n"), dict)
    assert isinstance(build_dual_face_shot_workflow("a", "b", "ma", "mb", "p", "n"), dict)
    assert isinstance(build_face_body_shot_workflow("f", "b", "p", "n"), dict)


# --- 双角色支持 (2026-07-23, 林夏 LoRA 流水线并入) ---

def test_both_characters_have_role_data():
    """xiaoya + linxia 双角色 5 镜 SHOTS/LINES/SPEAKER 都齐."""
    from lib.ep04_shots import (CHAR_BY_NAME, SHOTS_BY_NAME, LINES_BY_NAME,
                                I2V_MOTION_BY_NAME, SPEAKER_BY_NAME)
    for name in ("xiaoya", "linxia"):
        assert name in CHAR_BY_NAME
        assert set(SHOTS_BY_NAME[name].keys()) == {1, 2, 3, 4, 5}
        assert set(LINES_BY_NAME[name].keys()) == {1, 2, 3, 4, 5}
        assert set(I2V_MOTION_BY_NAME[name].keys()) == {1, 2, 3, 4, 5}
        assert set(SPEAKER_BY_NAME[name].keys()) == {1, 2, 3, 4, 5}


def test_linxia_clothes_lock_in_motion():
    """林夏 i2v motion 含全部 4 项衣服锁不变式 (ep03 实战教训)."""
    from lib.ep04_shots import I2V_MOTION_BY_NAME
    lock_phrases = [
        "misty-blue athletic top stays misty-blue",
        "white arc logo on the left chest stays visible",
        "black high-waist athletic pants stay black",
        "thin gray side stripe stays visible",
        "white running shoes stay white",
        "high ponytail stays high",
    ]
    for n in range(1, 6):
        motion = I2V_MOTION_BY_NAME["linxia"][n].lower()
        for phrase in lock_phrases:
            assert phrase.lower() in motion, f"shot{n} 缺衣服锁不变式: {phrase}"


def test_xiaoya_clothes_lock_in_motion():
    """小雅 i2v motion 含全部 2 项衣服锁不变式."""
    from lib.ep04_shots import I2V_MOTION_BY_NAME
    for n in range(1, 6):
        motion = I2V_MOTION_BY_NAME["xiaoya"][n].lower()
        assert "white office blouse stays white" in motion, f"shot{n} 缺衬衫锁"
        assert "pencil skirt stays the same" in motion, f"shot{n} 缺铅笔裙锁"


def test_refs_and_lora_name_per_character():
    """ablation 脚本按 character 切换 refs + LoRA 名."""
    import sys
    sys.path.insert(0, "../manju")
    sys.path.insert(0, "../manju/scripts")
    from scripts.gen_ep04_lora_ablation import REFS_BY_NAME, LORA_NAME_DEFAULT
    assert set(REFS_BY_NAME["xiaoya"].keys()) == {"face", "body"}
    assert set(REFS_BY_NAME["linxia"].keys()) == {"face", "body"}
    # LoadImage 节点期望 filename-only (ComfyUI/input/), 须 match ^[^/]+\.png$
    for name, refs in REFS_BY_NAME.items():
        for key, p in refs.items():
            import re
            assert re.match(r"^[\w\-]+\.png$", p), f"{name}/{key} 应是 filename-only, got: {p}"
    # LoRA 默认名匹配角色
    assert "xiaoya_lora" in LORA_NAME_DEFAULT["xiaoya"]
    assert "linxia_lora" in LORA_NAME_DEFAULT["linxia"]
    # trigger word 嵌入 prompt
    from scripts.gen_ep04_lora_ablation import build_positive
    assert ", xiaoya" in build_positive("xiaoya", 1, with_lora=True)
    assert ", linxia" in build_positive("linxia", 1, with_lora=True)
    # 不带 LoRA 时无 trigger word (避免空激活)
    assert ", xiaoya" not in build_positive("xiaoya", 1, with_lora=False)
    assert ", linxia" not in build_positive("linxia", 1, with_lora=False)


def test_lora_strength_per_character_default():
    """LoRA strength 0.85 对双角色都生效."""
    wf_xy = build_face_shot_workflow("f", "p", "n")
    wf_xy = _inject_lora(wf_xy, "xiaoya_lora.safetensors")
    wf_lx = build_face_shot_workflow("f", "p", "n")
    wf_lx = _inject_lora(wf_lx, "linxia_lora.safetensors")
    for wf, expect in ((wf_xy, "xiaoya_lora.safetensors"),
                       (wf_lx, "linxia_lora.safetensors")):
        lora = next(v for v in wf.values() if v["class_type"] == "LoraLoader")
        assert lora["inputs"]["lora_name"] == expect
        assert lora["inputs"]["strength_model"] == 0.85