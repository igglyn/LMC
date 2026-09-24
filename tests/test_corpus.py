import json

import pytest

from lmc.corpus import normalize_jsonl, read_jsonl, trace_to_pair
from lmc.schema import RawTrace


def test_extracts_explicit_thinking_window():
    trace = next(read_jsonl("examples/traces.jsonl"))
    pair = trace_to_pair(RawTrace.from_dict(trace))
    assert pair.thinking.startswith("Add the tens")
    assert pair.response == "42"
    assert pair.context[0].role == "user"


def test_normalize_rejects_duplicate_ids(tmp_path):
    trace = next(read_jsonl("examples/traces.jsonl"))
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join([json.dumps(trace), json.dumps(trace)]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate trace id"):
        normalize_jsonl(input_path, tmp_path / "output.jsonl")
