from copy import deepcopy

from envs.robomimic import RobomimicEnv as robomimic_env_module


def test_robosuite_15_metadata_is_translated_for_robosuite_14(monkeypatch) -> None:
    monkeypatch.setattr(robomimic_env_module.robosuite, "__version__", "1.4.1")
    metadata = {
        "env_name": "Lift",
        "env_kwargs": {
            "lite_physics": False,
            "controller_configs": {
                "type": "BASIC",
                "body_parts": {
                    "right": {
                        "type": "OSC_POSE",
                        "interpolation": None,
                        "input_ref_frame": "world",
                        "gripper": {"type": "GRIP"},
                    }
                },
            },
        },
    }
    original = deepcopy(metadata)

    compatible = robomimic_env_module._compatible_env_metadata(metadata)

    assert metadata == original
    assert "lite_physics" not in compatible["env_kwargs"]
    assert compatible["env_kwargs"]["controller_configs"] == {
        "type": "OSC_POSE",
        "interpolation": None,
    }


def test_robosuite_15_metadata_is_preserved_for_robosuite_15(monkeypatch) -> None:
    monkeypatch.setattr(robomimic_env_module.robosuite, "__version__", "1.5.1")
    metadata = {
        "env_kwargs": {
            "lite_physics": False,
            "controller_configs": {"type": "BASIC", "body_parts": {}},
        }
    }

    compatible = robomimic_env_module._compatible_env_metadata(metadata)

    assert compatible == metadata
    assert compatible is not metadata
