"""test_series_runner.py — run_series 多集批量编排核心单测."""
import pytest
from lib.series_runner import parse_eps


class TestParseEps:
    def test_single(self):
        assert parse_eps("20") == ["20"]

    def test_range(self):
        assert parse_eps("20-25") == ["20", "21", "22", "23", "24", "25"]

    def test_list(self):
        assert parse_eps("22,20,21") == ["20", "21", "22"]  # 排序去重

    def test_mixed(self):
        assert parse_eps("20-22,25") == ["20", "21", "22", "25"]

    def test_pad(self):
        assert parse_eps("3-4") == ["03", "04"]  # 与 EP_SCRIPTS 键规范一致

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="20-x"):
            parse_eps("20-x")
        with pytest.raises(ValueError):
            parse_eps("")
