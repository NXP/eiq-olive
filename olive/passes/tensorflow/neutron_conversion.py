#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#

import importlib
import logging
import os
import select
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Type

import billiard
from billiard.context import Process
from billiard.queues import Queue

from olive.common.utils import StrEnumBase
from olive.exception import OliveError
from olive.hardware import AcceleratorSpec
from olive.model.handler.tensorflow import TFLiteModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class NeutronConverterFlavors(StrEnumBase):
    """Flavors (versions) supported by NeutronConverterPass."""

    SDK_25_03 = "MCUXpresso SDK 25.03"
    SDK_25_06 = "MCUXpresso SDK 25.06"
    SDK_25_09 = "MCUXpresso SDK 25.09"
    SDK_25_12 = "MCUXpresso SDK 25.12"
    LF6_12_3_1_0_0 = "LF6.12.3_1.0.0"
    LF6_12_20_2_0_0 = "LF6.12.20_2.0.0"
    LF6_12_34_2_1_0 = "LF6.12.34_2.1.0"
    LF6_12_49_2_2_0 = "LF6.12.49_2.2.0"


class NeutronConverterTargets(StrEnumBase):
    """Targets supported by NeutronConverter."""

    IMXRT700 = "imxrt700"
    IMX943 = "imx943"
    IMX95 = "imx95"
    IMX952 = "imx952"
    MCXN54X = "mcxn54x"
    MCXN94X = "mcxn94x"
    S32K5 = "s32k5"
    S32N79 = "s32n79"


class NeutronConverterPassError(OliveError):
    """Error raised by NeutronConverterPass."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def load_neutron_converter(neutron_flavor):
    """Import correct version of neutron converter or raise exception when unknown."""
    flavor_mapping = {
        NeutronConverterFlavors.SDK_25_03: "neutron_converter_SDK_25_03",
        NeutronConverterFlavors.SDK_25_06: "neutron_converter_SDK_25_06",
        NeutronConverterFlavors.SDK_25_09: "neutron_converter_SDK_25_09",
        NeutronConverterFlavors.SDK_25_12: "eiq_neutron_sdk",
        NeutronConverterFlavors.LF6_12_3_1_0_0: "neutron_converter_SDK_25_03",
        NeutronConverterFlavors.LF6_12_20_2_0_0: "neutron_converter_SDK_25_06",
        NeutronConverterFlavors.LF6_12_34_2_1_0: "neutron_converter_SDK_25_09",
        NeutronConverterFlavors.LF6_12_49_2_2_0: "eiq_neutron_sdk",
    }

    module_name = flavor_mapping.get(neutron_flavor)
    if module_name is None:
        raise NeutronConverterPassError(f"Unsupported Neutron converter flavor: '{neutron_flavor}'.")

    return importlib.import_module(f"{module_name}.neutron_converter")


def _set_conversion_options(obj: object, config: dict[str, Any]):
    """Set conversion options from config dictionary."""
    for key, value in config.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        elif key not in ["target", "flavor"] and value is not None:
            message = f"This Neutron converter flavor does not support {key} option. Ignoring {key} option."
            logger.warning(message)


def convert_unsafe(tflite_model: bytes, config: dict[str, Any], result_queue: Queue):
    """Convert TFLite model with neutron converter.

    This function is intended to run in separate process.
    """
    neutron_target = config["target"]
    neutron_flavor = config["flavor"]

    if neutron_target not in list(NeutronConverterTargets):
        message = f"Unsupported neutron target: '{neutron_target}'."
        raise NeutronConverterPassError(message)

    neutron_converter = load_neutron_converter(neutron_flavor)

    cctx = neutron_converter.CompilationContext()
    cctx.targetOpts = neutron_converter.getNeutronTarget(neutron_target)

    compilation_opts = neutron_converter.CompilationOptions()
    _set_conversion_options(compilation_opts, config)

    cctx.compilationOpts = compilation_opts

    converted_model = neutron_converter.convertModel(list(tflite_model), cctx)
    converted_model = bytes(converted_model)

    result_queue.put(converted_model)


class CrashCapturingRunner:

    def run_with_crash_detection(self, target_function, tflite_model, config, result_queue):
        """Run the target function using multiprocessing with parameters."""
        # Create pipes for stdout/stderr redirection
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()

        def wrapper_function():
            """Wrap target_function and redirect its output."""
            # Redirect stdout and stderr to pipes
            os.dup2(stdout_write, 1)
            os.dup2(stderr_write, 2)

            # Close the read ends in the child process
            os.close(stdout_read)
            os.close(stderr_read)

            return target_function(tflite_model, config, result_queue)

        # Neutron converter is executed in separate process for two reasons:
        #  1. We are not able to import multiple neutron converter packages into single interpreter.
        #     There is problem with already "registered" class e.g. 'CompileOptions'.
        #  2. Neutron converter crashes whole interpreter in case of error during the conversion.
        process = Process(target=wrapper_function)
        process.start()

        # Close write ends in parent process
        os.close(stdout_write)
        os.close(stderr_write)

        # Monitor output in real-time
        def monitor_fd(pipe_fd, stream_name):
            try:
                while True:
                    # Wait for data from descriptor
                    ready, _, _ = select.select([pipe_fd], [], [], 0.1)

                    if ready:
                        data = os.read(pipe_fd, 4096)
                        if data and data != b"\n":
                            decoded_data = data.decode("utf-8", errors="replace").split("\n")
                            for line in decoded_data:
                                logger.info(line)
                        else:
                            break
                    elif not process.is_alive():
                        # Process already finished
                        break
            except Exception as e:
                message = f"Error monitoring '{stream_name}': {type(e)}"
                logger.exception(message)

        # Start stdout/stderr monitoring threads
        stdout_thread = threading.Thread(target=monitor_fd, args=(stdout_read, "stdout"))
        stderr_thread = threading.Thread(target=monitor_fd, args=(stderr_read, "stderr"))

        stdout_thread.daemon = True
        stderr_thread.daemon = True

        try:
            stdout_thread.start()
            stderr_thread.start()

            # Wait for neutron-converter to convert the model
            process.join(timeout=300)
            return_code = self.terminate_process_if_alive(process)

            # Wait for monitoring threads to finish
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
        finally:
            # Close read ends in parent process
            os.close(stdout_read)
            os.close(stderr_read)

        return return_code

    # noinspection PyMethodMayBeStatic
    def terminate_process_if_alive(self, process: Process) -> int:
        if process.is_alive():
            # Wait for conversion timeout -> process is still alive -> terminate
            process.terminate()
            process.join(timeout=5)

            return_code = -15  # TERM signal
        else:
            return_code = process.exitcode
        return return_code


class NeutronConversion(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "target": PassConfigParam(
                type_=NeutronConverterTargets,
                required=True,
                description=f"Target board, where converted model will be deployed. "
                f"Currently supported targets: [{', '.join(list(NeutronConverterTargets))}].",
                # Mention only those targets we support.
            ),
            "flavor": PassConfigParam(
                type_=NeutronConverterFlavors,
                required=True,
                description=f"Flavor (version) of Neutron converter used for the conversion. "
                f"Following flavors are currently supported: "
                f"[{', '.join(list(NeutronConverterFlavors))}].",
            ),
            "dumpHeaderFileInput": PassConfigParam(
                type_=bool,
                required=False,
                description="Option to export the input TensorFlowLite model as a header file.",
            ),
            "dumpHeaderFileOutput": PassConfigParam(
                type_=bool,
                required=False,
                description="Option to export the output TensorFlowLite model as a header file.",
            ),
            "useSequencer": PassConfigParam(
                type_=bool,
                required=False,
                description="Option to use the Neutron sequencer by generating Neutron bytecode. "
                "Note that this option cannot be used for Neutron-S targets (with subsystem).",
            ),
            "fetchConstantsToSRAM": PassConfigParam(
                type_=bool,
                required=False,
                description="Fetch constants (weights) from an external memory "
                "(external for Neutron, such as FLASH memory) into SRAM. "
                "This feature is relevant only for Neutron-C targets. "
                "This feature allows running models which do not fit into SRAM by offloading their weights to an"
                " external memory. Note that the weights prefetching will be done in parallel with the compute: "
                "while computing layer N the system will prefetch in parallel the weights for layer N+1. "
                "This ensures that the latency is optimal. For models that are I/O bound the time for prefetch"
                " might exceed the time for compute and so some extra penalty might occur. "
                "Therefore this feature must used only if needed: if model already fits into SRAM then it "
                "should be placed entirely into SRAM and used from there without using this feature.",
            ),
        }

    def _run_for_config(
        self,
        model: TFLiteModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> TFLiteModelHandler:
        if not isinstance(model, TFLiteModelHandler):
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")

        config = dict(config)

        output_model_path = Path(output_model_path)
        output_model_path.mkdir(parents=True, exist_ok=True)
        output_neutron_model_path = output_model_path / "neutron_model.tflite"

        # This ensures input and output paths for dumped header files are set to these locations
        config["input"] = str(model.model_path)
        config["output"] = str(output_neutron_model_path)

        # Read model as bytes.
        with Path.open(Path(model.model_path), "rb") as mp:
            tflite_model = mp.read()

        # noinspection PyUnresolvedReferences
        worker_queue = billiard.Manager().Queue()

        return_code = CrashCapturingRunner().run_with_crash_detection(
            convert_unsafe,
            tflite_model,
            config,
            worker_queue,
        )

        if return_code == 0:
            logger.info("Neutron Conversion completed successfully!")
        else:
            message = f"Neutron Conversion failed with return code: {return_code}"
            logger.error(message)

        if worker_queue.empty():
            raise NeutronConverterPassError("Neutron converter module terminated unexpectedly.")

        converted_model = worker_queue.get()

        with open(output_neutron_model_path, "wb") as f:
            f.write(converted_model)

        additional_files = []
        # We cannot directly set where this file is saved. It is saved in the same folder as input model
        # We copy it to output folder together with other artifacts
        if config["dumpHeaderFileInput"]:
            header_file_path = Path(model.model_path).with_suffix(".h")
            if header_file_path.exists():
                header_file_artifact_path = output_model_path / "neutron_input_model_exported.h"
                shutil.copy(header_file_path, header_file_artifact_path)
                header_file_path.unlink()

                additional_files.append(str(header_file_artifact_path))

        if config["dumpHeaderFileOutput"]:
            header_file_path = output_neutron_model_path.with_suffix(".h")
            if header_file_path.exists():
                additional_files.append(str(header_file_path))

        if additional_files:
            return TFLiteModelHandler(
                output_neutron_model_path, model_attributes={"additional_files": additional_files}
            )
        return TFLiteModelHandler(output_neutron_model_path)
