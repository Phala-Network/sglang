"""Run the shared parser error-path canaries on the selected source tree."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "parser_privacy_gate", ROOT / "scripts/phala/parser_privacy_gate.py"
)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_shared_parser_privacy_and_success_contracts(tmp_path):
    report = tmp_path / "parser-gate.json"
    with patch("sys.argv", ["parser_privacy_gate", "--report", str(report)]):
        assert GATE.main() == 0
    result = json.loads(report.read_text())
    assert result["privacy_passed"]
    assert result["positive_passed"]
    assert len(result["privacy_cases"]) == 15
    assert len(result["positive_cases"]) == 7
    assert len(result["behavior"]) == 126
