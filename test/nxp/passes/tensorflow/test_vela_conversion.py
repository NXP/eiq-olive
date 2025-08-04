#
# Copyright 2025 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.tensorflow.vela_conversion import VelaConversion

model_path = "test/nxp/resources/conv2d_model.tflite"
tflite_model_config = TFLiteModelHandler(model_path)


def test_vela_conversion_success(tmp_path):
    """Test successful run of VelaConversion pass. The pass configuration should be empty."""
    output_folder = str(tmp_path)
    output_folder = "/home/nxg05608/Documents"

    p = create_pass_from_dict(VelaConversion, {}, disable_search=True)

    p.run(tflite_model_config, output_folder)
