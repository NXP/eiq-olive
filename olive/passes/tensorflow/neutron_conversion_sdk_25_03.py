#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#

import logging
from pathlib import Path
from typing import Dict, Type

from olive.hardware import AcceleratorSpec
from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class NeutronConversionSDK2503(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "target": PassConfigParam(
                type_=str,
                required=True,
                description="Target board, where converted model will be deployed."
                "Currently supported target is 'imxrt700'.",  # Mention only those targets we support.
            ),
        }

    def get_neutron_targets(self) -> list[str]:
        """Return names of all neutron targets that converter supports right now."""
        return ["imxrt700", "mcxn54", "mcxn94x", "imx95", "imx943", "s32k5", "s32n79"]

    def _run_for_config(
        self,
        model: TFLiteModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> TFLiteModelHandler:

        if not isinstance(model, TFLiteModelHandler):
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")

        import neutron_converter_SDK_25_03.neutron_converter as neutron_converter  # noqa: PLC0415

        config = dict(config)
        neutron_target = config["target"]
        if neutron_target not in self.get_neutron_targets():
            raise ValueError(f"{neutron_target} is not valid neutron target.")

        # Read model as bytes.
        with Path.open(Path(model.model_path), "rb") as mp:
            model_data = mp.read()

        try:
            # Run conversion with profiling options.
            cctx = neutron_converter.CompilationContext()
            cctx.targetOpts = neutron_converter.getNeutronTarget(neutron_target)

            converted_model = neutron_converter.convertModel(list(model_data), cctx)
            converted_model = bytes(converted_model)
        except Exception as e:
            err_msg = f"NeutronConversion: {e}"
            logger.exception(err_msg)
            raise

        output_model_path = Path(output_model_path)
        output_model_path.mkdir(parents=True, exist_ok=True)
        output_model_path = output_model_path / "neutron_model.tflite"

        with open(output_model_path, "wb") as f:
            f.write(converted_model)

        return TFLiteModelHandler(output_model_path)
