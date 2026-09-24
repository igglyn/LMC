import json

import pytest

from lmc.config import ExperimentConfig


def test_loads_example_config():
    config = ExperimentConfig.from_path("configs/experiment.example.json")
    assert config.teacher.max_concurrent_requests == 8
    assert config.model.weights_path == "/models/LFM2.5-8B-A1B"
    assert config.thinking_window.preserve_kv_for_latent_steps


def test_rejects_window_that_leaks_across_boundaries():
    values = json.loads(open("configs/experiment.example.json", encoding="utf-8").read())
    values["thinking_window"]["clear_kv_at_window_end"] = False
    with pytest.raises(ValueError, match="clear K/V"):
        ExperimentConfig.from_dict(values)


def test_rejects_removed_external_teacher_endpoint_setting():
    values = json.loads(open("configs/experiment.example.json", encoding="utf-8").read())
    values["teacher"]["endpoint"] = "http://127.0.0.1:8000"
    with pytest.raises(TypeError, match="endpoint"):
        ExperimentConfig.from_dict(values)
