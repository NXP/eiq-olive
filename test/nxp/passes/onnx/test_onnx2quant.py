#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#

import os
import shutil
from pathlib import Path
from test.unit_test.utils import get_onnx_model_config

import numpy as np
import onnx
import pytest
from pydantic.v1 import ValidationError

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.nxp_onnx2quant import ONNX2Quant


class RandomCalibrationDataset:

    def __init__(self, shapes, input_names, path="tmp_calibration_dataset", np_type=np.float32, items_count=5):
        self.path = path
        self.shapes = shapes
        self.np_type = np_type
        self.items_count = items_count
        self.input_names = input_names

    def __enter__(self):

        Path.mkdir(self.path)

        for input_name, shape in zip(self.input_names, self.shapes):
            if os.path.exists(os.path.join(self.path, input_name)):
                raise Exception(f"Directory with name '{self.path}' already exists!")
            Path.mkdir(os.path.join(self.path, input_name))
            for x in range(self.items_count):
                input_vector = np.random.random(shape).reshape(shape).astype(self.np_type)

                np.save(os.path.join(self.path, input_name, f"{x}.npy"), input_vector)

        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path)


def create_onnx_model_with_multiple_inputs(path, input_name1="x", input_name2="y"):
    def get_onnx_model_multiple_inputs():
        x_shape = [2, 10]
        y_shape = [2, 10]

        # Define the graph
        graph = onnx.helper.make_graph(
            nodes=[
                onnx.helper.make_node("Add", [input_name1, input_name2], ["z"]),
                onnx.helper.make_node("Flatten", ["z"], ["output"]),
            ],
            name="graph-sub",
            inputs=[
                onnx.helper.make_tensor_value_info(input_name1, onnx.TensorProto.FLOAT, x_shape),
                onnx.helper.make_tensor_value_info(input_name2, onnx.TensorProto.FLOAT, y_shape),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ()),
            ],
        )

        return onnx.helper.make_model(graph)

    model = get_onnx_model_multiple_inputs()
    model_path = os.path.join(path, "multiple_inputs_model.onnx")
    onnx.save(model, model_path)
    return model_path


def test_onnx2quant_pass_no_config():
    """Test that pass without required parameters throws an error."""
    with pytest.raises(ValidationError):
        create_pass_from_dict(ONNX2Quant, {}, disable_search=True)


def test_onnx2quant_with_minimum_arguments(tmp_path):
    """Test the pass with only required argument."""
    input_model = get_onnx_model_config().create_model()
    config = {"calibration_dataset": "tmp_calibration_dataset", "allow_opset_10_and_lower": True}
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)
    with RandomCalibrationDataset([(1, 1)], ["input"]):
        output_folder = str(tmp_path)
        onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


def test_onnx2quant_all_parameters_defined(tmp_path):
    """Test the pass with all parameters defined."""
    input_model = get_onnx_model_config().create_model()
    config = {
        "calibration_dataset": "tmp_calibration_dataset",
        "allow_opset_10_and_lower": True,
        "per_channel": True,
        "symbolic_dimension_into_static": ["batch:1"],
        "set_input_shape": ["input:(1,1)"],
    }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)
    with RandomCalibrationDataset([(1, 1)], ["input"]):
        output_folder = str(tmp_path)
        onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


def test_onnx2quant_multiple_inputs_model(tmp_path):
    """Test the pass with model with multiple inputs."""
    model_path = create_onnx_model_with_multiple_inputs(tmp_path)
    input_model = get_onnx_model_config(model_path).create_model()
    config = {"calibration_dataset": "tmp_calibration_dataset", "allow_opset_10_and_lower": True}
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)
    with RandomCalibrationDataset([(2, 10), (2, 10)], ["x", "y"]):
        output_folder = str(tmp_path)
        onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()


@pytest.mark.parametrize("input_names", [
    ["hello,-.sdf$ßđˇ/", "y"],
    ["x", "."],
    ["", "abc"],
    ["111", ".."],
    ["x\t1", "y2"],
    ["x 1", "y\n2"]
])
def test_onnx2quant_bad_input_model_name(tmp_path, input_names):
    """Test that pass, where onnx model has invalid input names raises an error."""
    model_path = create_onnx_model_with_multiple_inputs(tmp_path, input_name1=input_names[0],
                                                        input_name2=input_names[1])
    input_model = get_onnx_model_config(model_path).create_model()
    config = {"calibration_dataset": "tmp_calibration_dataset", "allow_opset_10_and_lower": True}
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)
    output_folder = str(tmp_path)
    with pytest.raises(ValueError, match=r"cannot be used as a directory name for calibration dataset"):
        p.run(input_model, output_folder)


def test_onnx2quant_invalid_calibration_dataset_path(tmp_path):
    """Test that the pass with wrong calibration_dataset_mapping configuration raises an error."""
    input_model = get_onnx_model_config().create_model()
    config = {
        "calibration_dataset": "tmp_calibration",
        "allow_opset_10_and_lower": True,
    }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)

    with RandomCalibrationDataset([(1, 1)], ["input"]):
        output_folder = str(tmp_path)
        with pytest.raises(Exception, match=r"No such file or directory"):
            p.run(input_model, output_folder)


def test_onnx2quant_invalid_calibration_dataset_input_name(tmp_path):
    """Test that the pass with wrong input name in calibration dataset path raises an error."""
    input_model = get_onnx_model_config().create_model()
    config = {
        "calibration_dataset": "tmp_calibration_dataset",
        "allow_opset_10_and_lower": True,
    }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)

    with RandomCalibrationDataset([(1, 1)], ["x"]):
        output_folder = str(tmp_path)
        with pytest.raises(Exception, match=r"are missing from input feed"):
            p.run(input_model, output_folder)


def test_onnx2quant_invalid_symbolic_dim_into_static(tmp_path):
    """Test that the pass with wrong symbolic_dim_into_static configuration raises an error."""
    input_model = get_onnx_model_config().create_model()
    config = {
        "calibration_dataset": "tmp_calibration_dataset",
        "allow_opset_10_and_lower": True,
        "symbolic_dimension_into_static": ["batch:e"],
    }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)

    with RandomCalibrationDataset([(1, 1)], ["input"]):
        output_folder = str(tmp_path)
        with pytest.raises(Exception, match=r"in invalid format. Must be '<dim_name>:<dimension_size>'"):
            p.run(input_model, output_folder)


def test_onnx2quant_invalid_input_shape(tmp_path):
    """Test that the pass raises an exception when input shape is invalid."""
    input_model = get_onnx_model_config().create_model()
    config = {
        "calibration_dataset": "tmp_calibration_dataset",
        "allow_opset_10_and_lower": True,
        "set_input_shape": ["input:(a,b,c,d)"],
    }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)

    with RandomCalibrationDataset([(1, 1)], ["input"]):
        output_folder = str(tmp_path)
        with pytest.raises(Exception, match=r"in invalid format. Must be <dim_name>:\(<dim_0>,<dim_1>,...\)"):
            p.run(input_model, output_folder)
