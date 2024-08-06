#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
import logging
from pathlib import Path
from typing import Any, Dict

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from onnx2tflite.src.logger import conversion_log, MessageImportance

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

    def _log_conversion_logs(self):
        def _get_logging_fun(importance):
            if importance == MessageImportance.DEBUG:
                return logger.debug
            elif importance == MessageImportance.INFO:
                return logger.info
            elif importance == MessageImportance.WARNING:
                return logger.warning
            else:
                return logger.error

        def _parse_log(log_category, log: dict):
            # data = {
            #     "message": message,
            #     "logging_context_hierarchy": list(self._current_logging_context),
            #     "importance": importance.value,
            #     "message_id": self._log_count,
            # }
            return f'[TFLiteConversion:{log_category}] {log["message"]}'

        for log_category, logs in conversion_log.get_logs().items():
            for log in logs:
                fn = _get_logging_fun(MessageImportance(log["importance"]))
                fn(_parse_log(log_category, log))

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        data_root: str,
        config: Dict[str, Any],
        output_model_dir: str,
    ) -> TFLiteModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")

        from onnx2tflite.src.conversion_config import ConversionConfig
        from onnx2tflite.src.converter.convert import convert_model

        try:
            binary_tflite_model = convert_model(model.model_path, ConversionConfig(config))
        finally:
            self._log_conversion_logs()

        output_model_dir = Path(output_model_dir)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        output_model_path = output_model_dir / "model.tflite"

        with open(output_model_path, "wb") as f:
            f.write(binary_tflite_model)

        return TFLiteModelHandler(output_model_path)
