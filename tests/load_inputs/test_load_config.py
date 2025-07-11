'''
##############################################################
# Created Date: Friday, July 11th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import tempfile
import shutil
from typing import Any
import yaml
import pytest
from pathlib import Path
from realtwin.func_lib._b_load_inputs.loader_config import load_input_config


@pytest.fixture
def minimal_config_dict():
    return {
        "Network": {
            "NetworkVertices": [[-85.1, 35.0], [-85.2, 35.1], [-85.3, 35.2], [-85.4, 35.3]]
        },
        "Traffic": {},
        "Control": {}
    }


def write_yaml(tmp_path, data, filename="config.yaml"):
    config_path = tmp_path / filename
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return str(config_path)


def test_file_not_found(tmp_path: Path):
    non_existent = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_input_config(str(non_existent))


def test_not_yaml_file(tmp_path: Path, minimal_config_dict: dict):
    config_path = write_yaml(tmp_path, minimal_config_dict, "config.txt")
    with pytest.raises(ValueError):
        load_input_config(config_path)


def test_load_minimal_config(tmp_path: Path, minimal_config_dict: dict):
    config_path = write_yaml(tmp_path, minimal_config_dict)
    config = load_input_config(config_path)
    assert isinstance(config, dict)
    assert "input_dir" in config
    assert "output_dir" in config
    assert "Net_BBox" in config["Network"]


def test_vertices_invalid_list(tmp_path: Path, minimal_config_dict: dict[str, Any]):
    minimal_config_dict["Network"]["NetworkVertices"] = [1, 2, 3]
    config_path = write_yaml(tmp_path, minimal_config_dict)
    with pytest.raises(ValueError):
        load_input_config(config_path)


def test_vertices_invalid_type(tmp_path: Path, minimal_config_dict: dict[str, Any]):
    minimal_config_dict["Network"]["NetworkVertices"] = 12345
    config_path = write_yaml(tmp_path, minimal_config_dict)
    with pytest.raises(ValueError):
        load_input_config(config_path)


def test_missing_sections(tmp_path: Path, minimal_config_dict: dict[str, Any], capsys):
    config = minimal_config_dict.copy()
    del config["Traffic"]
    config_path = write_yaml(tmp_path, config)
    load_input_config(config_path)
    # Should log missing Traffic section, but not raise


def test_demo_data_not_string(tmp_path: Path, minimal_config_dict: dict[str, Any]):
    minimal_config_dict["demo_data"] = 123
    config_path = write_yaml(tmp_path, minimal_config_dict)
    config = load_input_config(config_path)
    assert config["demo_data"] is None


def test_demo_data_unavailable(tmp_path: Path, minimal_config_dict: dict[str, Any]):
    minimal_config_dict["demo_data"] = "notavailable"
    config_path = write_yaml(tmp_path, minimal_config_dict)
    config = load_input_config(config_path)
    assert config["demo_data"] is None
