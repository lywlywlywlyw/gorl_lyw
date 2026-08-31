from dataclasses import fields

import pytest

from flow_policy.config_utils import require_config_values
from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMConfig
from flow_policy.decoder_fm import DecoderFMConfig
from flow_policy.encoder_rlpd import EncoderConfig


@pytest.mark.parametrize(
    "config",
    (DecoderFMConfig(), Decoder1StepFMConfig(), EncoderConfig()),
)
def test_model_config_defaults_are_unspecified(config: object) -> None:
    assert all(getattr(config, field.name) is None for field in fields(config))


def test_missing_model_config_values_are_reported_together() -> None:
    with pytest.raises(ValueError, match="flow_steps.*learning_rate"):
        require_config_values(DecoderFMConfig())
