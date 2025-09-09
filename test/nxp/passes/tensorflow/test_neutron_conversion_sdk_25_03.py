#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#

from pathlib import Path

import pytest
from pydantic.v1 import ValidationError

from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.tensorflow.neutron_conversion_sdk_25_03 import NeutronConversionSDK2503

model_path = "test/nxp/resources/1.tflite"
tflite_model_config = TFLiteModelHandler(model_path)


def test_neutron_conversion_pass_no_config():
    """Test that pass without required parameters throws an error."""
    with pytest.raises(ValidationError):
        create_pass_from_dict(NeutronConversionSDK2503, {}, disable_search=True)


def test_neutron_conversion_success(tmp_path):
    """Test successful run of NeutronConversion pass."""
    pass_config = {"target": "imxrt700"}
    p = create_pass_from_dict(NeutronConversionSDK2503, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(tflite_model_config, output_folder)

    assert Path(neutron_model.model_path).exists()


def test_neutron_conversion_unsupported_target(tmp_path):
    """Test that providing wrong target raises an error."""
    pass_config = {"target": "my_favourite_target"}
    p = create_pass_from_dict(NeutronConversionSDK2503, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    with pytest.raises(ValueError, match=r"is not valid neutron target"):
        p.run(tflite_model_config, output_folder)
