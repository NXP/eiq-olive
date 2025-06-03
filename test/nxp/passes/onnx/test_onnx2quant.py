#
# Copyright 2025 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import pytest
import os
import numpy as np
import shutil
from pydantic.v1 import ValidationError
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.nxp_onnx2quant import ONNX2Quant
from test.unit_test.utils import get_onnx_model_config
from pathlib import Path

class randomCalibrationDataset:

    def __init__(self, shape, input_name, path="tmp_calibration_dataset", np_type=np.float32, items_count=5):
        self.path = path
        self.shape = shape
        self.np_type = np_type
        self.items_count = items_count
        self.input_name = input_name

    def __enter__(self):

        if os.path.exists(os.path.join(self.path, self.input_name)):
            raise Exception(f"Directory with name '{self.path}' already exists!")
        Path.mkdir(self.path)
        Path.mkdir(os.path.join(self.path, self.input_name))

        for x in range(self.items_count):
            input_vector = np.random.random(self.shape).reshape(self.shape).astype(self.np_type)
            np.save(os.path.join(self.path, self.input_name,  f"{x}.npy"), input_vector)

        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path)

def test_onnx2quant_pass_no_config():
    """Test that pass without required parameters throws an error."""
    with pytest.raises(ValidationError):
        create_pass_from_dict(ONNX2Quant, {}, disable_search=True)

def test_onnx2quant_with_minimum_arguments(tmp_path):
    """Test the pass with only required argument."""
    input_model = get_onnx_model_config().create_model()
    config = {
            "calibration_dataset": "tmp_calibration_dataset",
            "allow_opset_10_and_lower": True
        }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)
    with randomCalibrationDataset((1,1), "input"):
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
            "set_input_shape":["input:(1,1)"]
        }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)
    with randomCalibrationDataset((1,1), "input"):
        output_folder = str(tmp_path)
        onnx_model = p.run(input_model, output_folder)

    assert Path(onnx_model.model_path).exists()

def test_onnx2quant_invalid_calibration_dataset_path(tmp_path):
    """Test that the pass with wrong calibration_dataset_mapping configuration raises an error."""
    input_model = get_onnx_model_config().create_model()
    config = {
            "calibration_dataset": "tmp_calibration",
            "allow_opset_10_and_lower": True,
        }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)

    with randomCalibrationDataset((1,1), "input"):
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

    with randomCalibrationDataset((1,1), "x"):
        output_folder = str(tmp_path)
        with pytest.raises(Exception, match=r"are missing from input feed"):
            p.run(input_model, output_folder)

def test_onnx2quant_invalid_symbolic_dim_into_static(tmp_path):
    """Test that the pass with wrong symbolic_dim_into_static configuration raises an error."""
    input_model = get_onnx_model_config().create_model()
    config = {
            "calibration_dataset": "tmp_calibration_dataset",
            "allow_opset_10_and_lower": True,
            "symbolic_dimension_into_static": ["batch:e"]
        }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)

    with randomCalibrationDataset((1,1), "input"):
        output_folder = str(tmp_path)
        with pytest.raises(Exception, match=r"in invalid format. Must be '<dim_name>:<dimension_size>'"):
            p.run(input_model, output_folder)

def test_onnx2quant_invalid_input_shape(tmp_path):
    """Test that the pass raises an exception when input shape is invalid."""
    input_model = get_onnx_model_config().create_model()
    config = {
            "calibration_dataset": "tmp_calibration_dataset",
            "allow_opset_10_and_lower": True,
            "set_input_shape": ["input:(a,b,c,d)"]
        }
    p = create_pass_from_dict(ONNX2Quant, config, disable_search=True)

    with randomCalibrationDataset((1,1), "input"):
        output_folder = str(tmp_path)
        with pytest.raises(Exception, match=r"in invalid format. Must be <dim_name>:\(<dim_0>,<dim_1>,...\)"):
            p.run(input_model, output_folder)
