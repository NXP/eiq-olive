#
# Copyright 2024 NXP
#
# Licensed under the MIT License.
#
import logging
from pathlib import Path
from typing import Dict, Type, ClassVar

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from onnx2tflite.src.logger import conversion_log, MessageImportance

logger = logging.getLogger(__name__)


class TFLiteConversion(Pass):
    """Convert ONNX to TFLite model."""

    _default_config_params: ClassVar[dict[str, PassConfigParam]] = {
        "non_negative_indices": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Guarantee that an `indices` input tensor will always contain non-negative values."
        ),
        "cast_int64_to_int32": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Cast some nodes with type INT64 to INT32 when TFLite doesn't support INT64. Such nodes "
                        "are often used in ONNX to calculate shapes/indices, so full range of INT64 isn't "
                        "necessary."
        ),
        "accept_resize_rounding_error": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Accept the error caused by a different rounding approach of the ONNX `Resize` and "
                        "TFLite `ResizeNearestNeighbor` operators, and convert the model anyway."
        ),
        "ignore_opset_version": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Ignore the checks for supported opset versions of the ONNX model and try to convert it "
                        "anyway. This can result in an invalid output TFLite model."
        ),
        "allow_inputs_stripping": PassConfigParam(
            type_=bool,
            default_value=True,
            description="Model inputs will be removed if they are not necessary for inference and "
                        "their values are derived during the conversion."
        ),
        "keep_io_format": PassConfigParam(
            type_=bool,
            default_value=True,
            description="Keep the format of input and output tensors of the converted model the same, "
                        "as in the original ONNX model (NCHW)."
        ),
        "skip_shape_inference": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Shape inference will be skipped before model conversion. This option can "
                        "be used only if model's shapes are fully defined. Defined shapes are necessary for "
                        "successful conversion."
        ),
        "qdq_aware_conversion": PassConfigParam(
            type_=bool,
            default_value=True,
            description="Quantized QDQ model with QDQ pairs (Q-Ops created by QDQ quantizer) will be "
                        "converted into optimized variant with QDQ pairs represented as tensors' "
                        "quantization parameters."
        ),
        "symbolic_dimension_into_static": PassConfigParam(
            type_=list[str],
            default_value=[],
            description="Change symbolic dimension in model to static (fixed) value. Provided mapping must "
                        "follow this format '<dim_name>:<dim_size>', for example 'batch:1'. Multiple mappings "
                        "can be specified."
        ),
        "set_input_shape": PassConfigParam(
            type_=list[str],
            default_value=[],
            description="Override model input shape. Provided mapping must follow format '<dim_name>:(<dim_0>,"
                        "<dim_1>,...)', for example 'input_1:(1,3,224,224)'. Shapes of multiple inputs can be"
                        "specified."
        ),
        "dont_skip_nodes_with_known_outputs": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Sometimes it is possible to statically infer the output data of some nodes. These nodes "
                        "will then not be a part of the output model. This flag will force the converter to keep "
                        "them in anyway."
        ),
        "allow_select_ops": PassConfigParam(
            type_=bool,
            default_value=True,
            description="Allow the converter to use the `SELECT_TF_OPS` operators, which require Flex delegate at "
                        "runtime."
        ),
    }

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return cls._default_config_params

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: Type[BasePassConfig],
        output_model_dir: str,
    ) -> TFLiteModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")

        from onnx2tflite.src.conversion_config import ConversionConfig  # noqa: PLC0415
        import onnx2tflite.src.converter.convert as convert  # noqa: PLC0415

        config = dict(config)

        if "symbolic_dimension_into_static" in config:
            config["symbolic_dimensions_mapping"] = convert.parse_symbolic_dimensions_mapping(
                config["symbolic_dimension_into_static"]
            )

        if "set_input_shape" in config:
            config["input_shapes_mapping"] = convert.parse_input_shape_mapping(config["set_input_shape"])

        binary_tflite_model = convert.convert_model(model.model_path, ConversionConfig(config))

        output_model_dir = Path(output_model_dir)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        output_model_path = output_model_dir / "model.tflite"

        with open(output_model_path, "wb") as f:
            f.write(binary_tflite_model)

        return TFLiteModelHandler(output_model_path)
