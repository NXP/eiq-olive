#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#
import os
import pathlib
from pathlib import Path

import pytest
from pydantic.v1 import ValidationError

from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.tensorflow.neutron_conversion import NeutronConversion

_NXP_TEST_DIR = pathlib.Path(__file__).parent.parent.parent
model_path = os.path.join(_NXP_TEST_DIR, "resources", "1.tflite")


def test_neutron_conversion_pass_no_config():
    """Test that pass without required parameters throws an error."""
    with pytest.raises(ValidationError):
        create_pass_from_dict(NeutronConversion, {}, disable_search=True)


def test_neutron_conversion_sdk_25_03_success(tmp_path):
    """Test successful run of NeutronConversion pass. SDK 25.03."""
    pass_config = {"target": "imxrt700", "flavor": "MCUXpresso SDK 25.03"}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(TFLiteModelHandler(model_path), output_folder)

    assert Path(neutron_model.model_path).exists()


def test_neutron_conversion_sdk_25_06_success(tmp_path):
    """Test successful run of NeutronConversion pass for SDK 25.06."""
    pass_config = {"target": "imxrt700", "flavor": "MCUXpresso SDK 25.06"}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(TFLiteModelHandler(model_path), output_folder)

    assert Path(neutron_model.model_path).exists()

@pytest.mark.parametrize("target", ["imxrt700", "imx95"])
def test_neutron_conversion_sdk_25_09_success(tmp_path, target):
    """Test successful run of NeutronConversion pass for SDK 25.09."""
    pass_config = {"target": target, "flavor": "MCUXpresso SDK 25.09"}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(TFLiteModelHandler(model_path), output_folder)

    assert Path(neutron_model.model_path).exists()


def test_neutron_conversion_unsupported_target(tmp_path):
    """Test that providing wrong target raises an error."""
    pass_config = {"target": "my_favourite_target", "flavor": "MCUXpresso SDK 25.03"}
    with pytest.raises(ValidationError):
        _ = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
