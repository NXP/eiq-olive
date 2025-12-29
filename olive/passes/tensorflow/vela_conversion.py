#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#

import logging
import shutil
from pathlib import Path
from typing import Dict, Type

from olive.hardware import AcceleratorSpec
from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class VelaConversion(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {}

    def _run_for_config(
        self,
        model: TFLiteModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> TFLiteModelHandler:
        if not isinstance(model, TFLiteModelHandler):
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")

        # Conversion:
        from ethosu.vela import model_reader, vela  # noqa: PLC0415
        from ethosu.vela.architecture_features import ArchitectureFeatures  # noqa: PLC0415
        from ethosu.vela.compiler_driver import CompilerOptions  # noqa: PLC0415
        from ethosu.vela.hillclimb_allocation import HillClimbAllocator  # noqa: PLC0415
        from ethosu.vela.nn_graph import TensorAllocator  # noqa: PLC0415
        from ethosu.vela.scheduler import OptimizationStrategy, SchedulerOptions  # noqa: PLC0415
        from ethosu.vela.tensor import Tensor  # noqa: PLC0415

        vela_ini_path = Path(__file__).parent / "misc/vela.ini"

        try:
            arch = ArchitectureFeatures(
                vela_config_files=vela_ini_path,
                system_config="Ethos_U65_High_End",
                # Auzone configuration see https://bitbucket.sw.nxp.com/projects/AITEC/repos/auzone-runtime-modelrunner/browse/profiling/vela_offline/vela_offline_profiling.cc
                memory_mode="Sram_Only",
                accelerator_config="ethos-u65-256",
                # Auzone configuration https://bitbucket.sw.nxp.com/projects/AITEC/repos/auzone-runtime-modelrunner/browse/profiling/vela_offline/vela_offline_profiling.cc
                max_blockdep=ArchitectureFeatures.MAX_BLOCKDEP,
                verbose_config=False,
                arena_cache_size=384 * 1024,
            )

            # Vela memory mode cannot be set to default value together with vela_config_files.
            # First set it to some value and then set the default.
            arch._set_default_mem_mode()
            vela_output_dir = Path("./output")

            compiler_options = CompilerOptions(
                tensor_allocator=TensorAllocator.HillClimb,
                output_dir=str(vela_output_dir),
                hillclimb_max_iterations=HillClimbAllocator.MAX_ITERATIONS,
                cpu_tensor_alignment=Tensor.AllocationQuantum,
            )

            scheduler_options = SchedulerOptions(
                optimization_strategy=OptimizationStrategy.Performance,
                sram_target=arch.arena_cache_size,
                verbose_schedule=False,
                verbose_progress=True,
            )

            vela.process(
                input_name=model.model_path,
                enable_debug_db=None,
                arch=arch,
                model_reader_options=model_reader.ModelReaderOptions(),
                compiler_options=compiler_options,
                scheduler_options=scheduler_options,
                output_format="tflite",
            )

            # Copy optimized model from the Vela output dir to the specified dir:
            model_name = Path(model.model_path).stem + "_vela.tflite"

            output_model_path = Path(output_model_path)
            output_model_path.mkdir(parents=True, exist_ok=True)

            shutil.copy(src=str(vela_output_dir / model_name), dst=str(output_model_path / model_name))

            # Remove Vela outputs:
            shutil.rmtree(vela_output_dir)

        except Exception as e:
            err_msg = f"VelaConversion failed: {e}"
            logger.exception(err_msg)
            raise

        return TFLiteModelHandler(output_model_path / model_name)
