#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
from pathlib import Path

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.tensorflow.conversion import TFLiteConversion
from unit_test.utils import get_onnx_model_config


def test_tflite_conversion_pass__no_config(tmp_path):
    input_model = get_onnx_model_config().create_model()
    p = create_pass_from_dict(TFLiteConversion, {}, disable_search=True)
    output_folder = str(tmp_path)
    onnx_model = p.run(input_model, None, output_folder)

    assert Path(onnx_model.model_path).exists()


def test_tflite_conversion_pass__select_ops_enabled(tmp_path):
    input_model = get_onnx_model_config().create_model()
    config = {
        "allow_select_ops": True
    }
    p = create_pass_from_dict(TFLiteConversion, config, disable_search=True)
    output_folder = str(tmp_path)
    onnx_model = p.run(input_model, None, output_folder)

    assert Path(onnx_model.model_path).exists()
