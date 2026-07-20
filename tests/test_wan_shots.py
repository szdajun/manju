"""lib/wan_shots 不变量(单 5B, 无模型分工)。纯 python, manju venv 可跑。"""
from lib.wan_shots import WAN_PROMPTS, WAN_NEG


def test_all_10_shots_have_motion_prompt():
    for s in range(1, 11):
        assert s in WAN_PROMPTS, f"shot {s} 缺运动 prompt"
        assert len(WAN_PROMPTS[s]) > 20, f"shot {s} prompt 太短(应描述连续运动)"


def test_negative_prompt_nonempty():
    assert WAN_NEG and len(WAN_NEG) > 20


def test_no_model_split_symbols():
    """2026-07-20 改单 5B: 模块不应再保留 14B/1.3B 分工符号。"""
    import lib.wan_shots as m
    for sym in ("SHOTS_14B", "SHOTS_13B", "MODEL_FOR_SHOT"):
        assert not hasattr(m, sym), f"已废弃符号仍存在: {sym}"
