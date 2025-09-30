#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#
import os.path
import pathlib
from pathlib import Path

import pytest

from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.tensorflow.vela_conversion import VelaConversion

_NXP_TEST_DIR = pathlib.Path(__file__).parent.parent.parent
model_path = os.path.join(_NXP_TEST_DIR, "resources", "conv2d_model.tflite")
tflite_model_config = TFLiteModelHandler(model_path)


def test_vela_conversion_success(tmp_path):
    """Test successful run of VelaConversion pass. The pass configuration should be empty."""
    output_folder = str(tmp_path)

    p = create_pass_from_dict(VelaConversion, {}, disable_search=True)

    vela_model = p.run(tflite_model_config, output_folder)
    assert Path(vela_model.model_path).exists()
    assert Path(vela_model.model_path).is_file()


def test_vela_conversion_file_not_exists(tmp_path):
    """Test Vela conversion pass raises exception when provided non existing model."""
    output_folder = str(tmp_path)
    non_existing_model = TFLiteModelHandler("model_non_existing.tflite")

    p = create_pass_from_dict(VelaConversion, {}, disable_search=True)

    with pytest.raises(FileNotFoundError, match=r"Model path model_non_existing.tflite does not exist."):
        p.run(non_existing_model, output_folder)
