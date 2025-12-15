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
from olive.passes.tensorflow.neutron_conversion import (
    NeutronConversion,
    NeutronConverterPassError,
    NeutronConverterTargets,
)

_NXP_TEST_DIR = pathlib.Path(__file__).parent.parent.parent
model_path = os.path.join(_NXP_TEST_DIR, "resources", "1.tflite")


def test_neutron_conversion_pass_no_config():
    """Test that pass without required parameters throws an error."""
    with pytest.raises(ValidationError):
        create_pass_from_dict(NeutronConversion, {}, disable_search=True)


@pytest.mark.parametrize(
    "neutron_flavor",
    [
        "MCUXpresso SDK 25.03",
        "MCUXpresso SDK 25.06",
        "MCUXpresso SDK 25.09",
        "MCUXpresso SDK 25.12",
        "LF6.12.3_1.0.0",
        "LF6.12.20_2.0.0",
        "LF6.12.34_2.1.0",
        "LF6.12.49_2.2.0",
    ],
)
def test_neutron_conversion_success(tmp_path, neutron_flavor):
    """Test successful run of NeutronConversion pass. SDK 25.03."""
    pass_config = {"target": "imxrt700", "flavor": neutron_flavor}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(TFLiteModelHandler(model_path), output_folder)

    assert Path(neutron_model.model_path).exists()


@pytest.mark.parametrize("target", [t.value for t in NeutronConverterTargets])
def test_neutron_conversion_sdk_25_09_success(tmp_path, target):
    """Test successful run of NeutronConversion pass for SDK 25.09."""
    pass_config = {"target": target, "flavor": "MCUXpresso SDK 25.09"}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(TFLiteModelHandler(model_path), output_folder)

    assert Path(neutron_model.model_path).exists()


@pytest.mark.parametrize("target", [t.value for t in NeutronConverterTargets])
def test_neutron_conversion_sdk_25_12_success(tmp_path, target):
    """Test successful run of NeutronConversion pass for SDK 25.12."""
    pass_config = {"target": target, "flavor": "MCUXpresso SDK 25.12"}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(TFLiteModelHandler(model_path), output_folder)

    assert Path(neutron_model.model_path).exists()


def test_neutron_conversion_unsupported_target(tmp_path):
    """Test that providing wrong target raises an error."""
    pass_config = {"target": "my_favourite_target", "flavor": "MCUXpresso SDK 25.03"}
    with pytest.raises(ValidationError):
        _ = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)


def test_neutron_conversion_unsupported_flavor(tmp_path):
    """Test that providing wrong flavor raises an error."""
    pass_config = {"target": "imxrt700", "flavor": "MCUXpresso SDK XXX"}
    with pytest.raises(ValidationError):
        _ = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)


@pytest.mark.parametrize(
    "parameters",
    [
        {"target": "imx95", "flavor": "MCUXpresso SDK 25.09", "useSequencer": True},
        {"target": "imx95", "flavor": "MCUXpresso SDK 25.09", "fetchConstantsToSRAM": True},
    ],
)
def test_neutron_conversion_wrong_parameter_combination(tmp_path, parameters):
    """Test that parameters that are for neutron C cannot be used with neutron S target."""
    p = create_pass_from_dict(NeutronConversion, parameters, disable_search=True)
    output_folder = str(tmp_path)
    with pytest.raises(NeutronConverterPassError):
        p.run(TFLiteModelHandler(model_path), output_folder)


@pytest.mark.parametrize(
    "parameters",
    [
        {"target": "imxrt700", "flavor": "MCUXpresso SDK 25.09", "useSequencer": True},
        {"target": "imxrt700", "flavor": "MCUXpresso SDK 25.09", "fetchConstantsToSRAM": True},
    ],
)
def test_neutron_conversion_correct_parameter_combination(tmp_path, parameters):
    """Test that parameters that are for neutron C can be correctly used for neutron C target."""
    p = create_pass_from_dict(NeutronConversion, parameters, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(TFLiteModelHandler(model_path), output_folder)
    assert Path(neutron_model.model_path).exists()
