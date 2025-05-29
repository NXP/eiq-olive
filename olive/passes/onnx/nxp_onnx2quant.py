#
# Copyright 2025 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from typing import Dict, Type
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from pathlib import Path
import onnx
import logging

from onnx2tflite.src.logger import conversion_log, MessageImportance
from onnx2quant.__main__ import NpyCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2quant.qdq_quantization import QDQQuantizer
import onnx2tflite.src.converter.convert as convert
from olive.common.config_utils import ParamCategory

logger = logging.getLogger(__name__)

class ONNX2Quant(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
        "per_channel": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Quantize some weight tensors per-channel instead of per-tensor. This should result in a "
                        "higher accuracy."
        ),
        "allow_opset_10_and_lower": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Allow quantization of models with opset version 10 and lower. Quantization of such models"
                        "can produce invalid models because opset is forcefully updated to version 11. This applies "
                        "especially to models with operators: Clip, Dropout, BatchNormalization and Split."
        ),
        "calibration_dataset": PassConfigParam(
            type_=str,
            category= ParamCategory.PATH,
            required=True,
            description="Path to a calibration dataset directory. The directory should contain subdirectories "
                        "for each model input. Each subdirectory then contains *.npy files."
        ),
        "symbolic_dimension_into_static": PassConfigParam(
            type_=list[str],
            default_value=[],
            description="Change symbolic dimension in model to static (fixed) value. Provided mapping must "
                        "follow this format '<dim_name>:<dim_size>', for example 'batch:1'."
        ),
        "set_input_shape": PassConfigParam(
            type_=list[str],
            default_value=[],
            description="Override model input shape. Provided mapping must follow format '<dim_name>:(<dim_0>,"
                        "<dim_1>,...)', for example 'input_1:(1,3,224,224)'."
        )
    }

    def _log_quantization_logs(self):
        def _get_logging_fn(importance):
            severity_map = {
                MessageImportance.DEBUG: logger.debug,
                MessageImportance.INFO: logger.info,
                MessageImportance.WARNING: logger.warning
            }
            return severity_map.get(importance, logger.error)

        def _parse_log(log_category, log: dict):
            # Log dictionary data:
            # data = {
            #     "message": message,
            #     "logging_context_hierarchy": list(self._current_logging_context),
            #     "importance": importance.value,
            #     "message_id": self._log_count,
            # }
            return f'[ONNX2Quant:{log_category}] {log["message"]}'

        for log_category, logs in conversion_log.get_logs().items():
            for log in logs:
                fn = _get_logging_fn(MessageImportance(log["importance"]))
                fn(_parse_log(log_category, log))

    def _create_calibration_dataset_mapping(self, dataset_path: Path) -> dict[str,str]:
        return {directory.name : str(directory) for directory in dataset_path.iterdir()
                if directory.is_dir()}


    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: Type[BasePassConfig],
        output_model_dir: str,
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")

        config = dict(config)
        try:
            if "symbolic_dimension_into_static" in config:
                    config["symbolic_dimensions_mapping"] = convert.parse_symbolic_dimensions_mapping(
                        config["symbolic_dimension_into_static"]
                    )

            if "set_input_shape" in config:
                config["input_shapes_mapping"] = convert.parse_input_shape_mapping(config["set_input_shape"])


            calibration_dataset_mapping = self._create_calibration_dataset_mapping(Path(config["calibration_dataset"]))
            print(calibration_dataset_mapping)
            calibration_data_reader = NpyCalibrationDataReader(calibration_dataset_mapping)
            quantization_config = QuantizationConfig(calibration_data_reader, config)
            onnx_model = onnx.load(model.model_path)
            quantized_model = QDQQuantizer().quantize_model(onnx_model, quantization_config=quantization_config)
        finally:
            self._log_quantization_logs()

        output_model_dir = Path(output_model_dir)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        output_model_path = output_model_dir / "model_quantized.onnx"

        onnx.save_model(quantized_model, output_model_path)

        return ONNXModelHandler(output_model_path)