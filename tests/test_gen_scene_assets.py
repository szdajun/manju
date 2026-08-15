"""gen_scene_assets 场景资产单测 (prompt/场景提取/workflow)."""
from scripts import gen_scene_assets as gsa
from lib.comfyui_client import build_face_body_shot_workflow


def test_build_scene_prompt_empty_no_people():
    p = gsa.build_scene_prompt("现代健身房")
    assert "现代健身房" in p
    assert "no people" in p and "no person" in p
    assert "wide angle" in p


def test_build_scene_prompt_empty_desc():
    assert "no people" in gsa.build_scene_prompt("")


def test_scene_key():
    assert gsa._scene_key("现代明亮健身房 interior") != ""
    assert gsa._scene_key("跑步机区") != ""


def test_scenes_from_script_field():
    script = {"scenes": ["健身房", "卧室"], "shots": {}}
    assert gsa._scenes_from_script(script) == ["健身房", "卧室"]


def test_scenes_fallback_from_shots():
    script = {"shots": {
        "1": {"shot": "medium shot, modern gym interior with treadmill, bright"},
        "2": {"shot": "full shot, living room background with sofa"},
    }}
    scenes = gsa._scenes_from_script(script)
    assert any("interior" in s for s in scenes)
    assert any("background" in s or "room" in s for s in scenes)


def test_scene_workflow_adds_third_adapter():
    """scene_ref_name → 加 ref_scene/unified_scene/ipa_scene, ksampler 走 ipa_scene."""
    wf = build_face_body_shot_workflow(
        "f.png", "b.png", "pos", "neg", scene_ref_name="scene.png", weight_scene=0.5)
    assert "ref_scene" in wf and "unified_scene" in wf and "ipa_scene" in wf
    assert wf["ipa_scene"]["inputs"]["weight"] == 0.5
    assert wf["unified_scene"]["inputs"]["preset"] == "PLUS (high strength)"
    ks = [v for v in wf.values() if v["class_type"] == "KSampler"][0]
    assert ks["inputs"]["model"] == ["ipa_scene", 0]


def test_scene_workflow_without_scene_keeps_original():
    wf = build_face_body_shot_workflow("f.png", "b.png", "pos", "neg")
    assert "ref_scene" not in wf
    ks = [v for v in wf.values() if v["class_type"] == "KSampler"][0]
    assert ks["inputs"]["model"] == ["ipa_body", 0]
