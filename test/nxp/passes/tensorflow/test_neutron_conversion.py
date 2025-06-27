from pathlib import Path
from pkgutil import ModuleInfo
from unittest.mock import patch

import pytest
from pydantic.v1 import ValidationError

from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.tensorflow.neutron_conversion import NeutronConversion

model_path = "test/nxp/resources/1.tflite"
tflite_model_config = TFLiteModelHandler(model_path)


def test_neutron_conversion_pass_no_config():
    """Test that pass without required parameters throws an error."""
    with pytest.raises(ValidationError):
        create_pass_from_dict(NeutronConversion, {}, disable_search=True)


def test_neutron_conversion_success(tmp_path):
    """Test successful run of NeutronConversion pass."""
    pass_config = {"target": "imxrt700"}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    neutron_model = p.run(tflite_model_config, output_folder)

    assert Path(neutron_model.model_path).exists()


def test_neutron_conversion_unsupported_target(tmp_path):
    """Test that providing wrong target raises an error."""
    pass_config = {"target": "my_favourite_target"}
    p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
    output_folder = str(tmp_path)
    with pytest.raises(ValueError, match=r"is not valid neutron target"):
        p.run(tflite_model_config, output_folder)


def test_neutron_conversion_no_converter_installed(tmp_path):
    """Test that if no neutron converter module is installed, pass will raise an error."""
    fake_installed_modules = [ModuleInfo(None, "MyModule1", False), ModuleInfo(None, "MyModule2", False)]
    with patch("pkgutil.iter_modules", return_value=fake_installed_modules):
        pass_config = {"target": "imx95"}
        p = create_pass_from_dict(NeutronConversion, pass_config, disable_search=True)
        output_folder = str(tmp_path)
        with pytest.raises(ImportError, match=r"No neutron_converter module installed."):
            p.run(tflite_model_config, output_folder)
