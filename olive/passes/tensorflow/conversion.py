#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
from pathlib import Path
from typing import Any, Dict

from olive import logging
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class TFLiteConversion(Pass):
    """Convert ONNX to TFLite model."""

    _default_config_params = {
        "allow_select_ops": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Allow the converter to use the `SELECT_TF_OPS` operators, which require Flex delegate at "
                        "runtime."
        )
    }

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return cls._default_config_params

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> TFLiteModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")

        from onnx2tflite.src.conversion_config import ConversionConfig
        from onnx2tflite.src.converter.convert import convert_model

        binary_tflite_model = convert_model(model.model_path, ConversionConfig(config))

        output_model_path = Path(output_model_path) / "model.tflite"

        with open(output_model_path, "wb") as f:
            f.write(binary_tflite_model)

        return TFLiteModelHandler(output_model_path)
