#
# Copyright 2025 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Type

from olive.hardware import AcceleratorSpec
from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class NeutronConversion(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "target": PassConfigParam(
                type_=str,
                required=True,
                description="Target board, where converted model will be deployed."
                "Currently supported targets are 'imxrt700', 'imx95'.",  # Mention only those targets we support.
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

        config = dict(config)
        neutron_target = config["target"]
        if neutron_target not in self.get_neutron_targets():
            raise ValueError(f"{neutron_target} is not valid neutron target.")

        neutron_converter_modules = [
            module.name for module in pkgutil.iter_modules() if module.name.startswith("neutron_converter")
        ]

        if len(neutron_converter_modules) == 0:
            logger.error("NeutronConverter: No neutron_converter module installed.")
            raise ImportError("NeutronConverter: No neutron_converter module installed.")
        neutron_converter = importlib.import_module(f"{neutron_converter_modules[0]}.neutron_converter")

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
