#
# Copyright 2024 NXP
#
# Licensed under the MIT License.
#
from pathlib import Path
from test.unit_test.utils import get_onnx_model_config

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.tensorflow.conversion import TFLiteConversion


def test_tflite_conversion_pass__no_config(tmp_path):
    input_model = get_onnx_model_config().create_model()
    p = create_pass_from_dict(TFLiteConversion, {}, disable_search=True)
    output_folder = str(tmp_path)
    onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


def test_tflite_conversion_pass__select_ops_enabled(tmp_path):
    input_model = get_onnx_model_config().create_model()
    config = {"allow_select_ops": True}
    p = create_pass_from_dict(TFLiteConversion, config, disable_search=True)
    output_folder = str(tmp_path)
    onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


def test_tflite_conversion_pass__define_all_params(tmp_path):
    input_model = get_onnx_model_config().create_model()
    config = {
        "non_negative_indices": True,
        "cast_int64_to_int32": True,
        "accept_resize_rounding_error": True,
        "ignore_opset_version": True,
        "allow_inputs_stripping": True,
        "keep_io_format": True,
        "skip_shape_inference": False,
        "qdq_aware_conversion": True,
        "symbolic_dimension_into_static": ["batch:1"],
        "set_input_shape": ["input:(1,1)"],
        "dont_skip_nodes_with_known_outputs": True,
        "allow_select_ops": True,
    }
    p = create_pass_from_dict(TFLiteConversion, config, disable_search=True)
    output_folder = str(tmp_path)
    onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


def test_tflite_conversion_pass__invalid_shape_definition(tmp_path):
    input_model = get_onnx_model_config().create_model()
    config = {
        "set_input_shape": ["input::::(1,2,3)"],
    }
    p = create_pass_from_dict(TFLiteConversion, config, disable_search=True)
    output_folder = str(tmp_path)

    with pytest.raises(Exception) as e:  # noqa: PT011
        p.run(input_model, output_folder)

    assert "in invalid format. Must be <dim_name>:(<dim_0>,<dim_1>,...)" in str(e)


def test_tflite_conversion_pass__symbolic_dim_definition(tmp_path):
    input_model = get_onnx_model_config().create_model()
    config = {
        "symbolic_dimension_into_static": ["batch:a"],
    }
    p = create_pass_from_dict(TFLiteConversion, config, disable_search=True)
    output_folder = str(tmp_path)

    with pytest.raises(Exception) as e:  # noqa: PT011
        p.run(input_model, output_folder)

    assert "in invalid format. Must be '<dim_name>:<dimension_size>'" in str(e)
