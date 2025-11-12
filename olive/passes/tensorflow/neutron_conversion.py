#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#

import importlib
import logging
import os
import select
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
    LF6_12_3_1_0_0 = "LF6.12.3_1.0.0"
    LF6_12_20_2_0_0 = "LF6.12.20_2.0.0"
    LF6_12_34_2_1_0 = "LF6.12.34_2.1.0"


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
        NeutronConverterFlavors.LF6_12_3_1_0_0: "neutron_converter_SDK_25_03",
        NeutronConverterFlavors.LF6_12_20_2_0_0: "neutron_converter_SDK_25_06",
        NeutronConverterFlavors.LF6_12_34_2_1_0: "neutron_converter_SDK_25_09",
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
        elif key not in ["target", "flavor"]:
            message = f"This Neutron converter flavor does not support {key} option. Ignoring {key} option."
            logger.warning(message)


def convert_unsafe(tflite_model: bytes, model_path: str, config: dict[str, Any], result_queue: Queue):
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

    # Set path to output model to compilation options as well,
    # so the exported header files are in the same directory and with the same name as converted model
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    model_path = model_path / "neutron_model.tflite"

    compilation_opts.output = str(model_path)

    cctx.compilationOpts = compilation_opts

    converted_model = neutron_converter.convertModel(list(tflite_model), cctx)
    converted_model = bytes(converted_model)

    result_queue.put(converted_model)


class CrashCapturingRunner:

    def run_with_crash_detection(self, target_function, tflite_model, model_path, config, result_queue):
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

            return target_function(tflite_model, model_path, config, result_queue)

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
                default=False,
                description="Option to export the input TensorFlowLite model as a header file.",
            ),
            "dumpHeaderFileOutput": PassConfigParam(
                type_=bool,
                required=False,
                default=False,
                description="Option to export the output TensorFlowLite model as a header file.",
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

        # Read model as bytes.
        with Path.open(Path(model.model_path), "rb") as mp:
            tflite_model = mp.read()

        # noinspection PyUnresolvedReferences
        worker_queue = billiard.Manager().Queue()

        return_code = CrashCapturingRunner().run_with_crash_detection(
            convert_unsafe,
            tflite_model,
            output_model_path,
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

        output_model_path = Path(output_model_path)
        output_model_path.mkdir(parents=True, exist_ok=True)
        output_model_path = output_model_path / "neutron_model.tflite"

        with open(output_model_path, "wb") as f:
            f.write(converted_model)

        if config["dumpHeaderFileInput"] or config["dumpHeaderFileOutput"]:
            additional_files = {"additional_files": [str(output_model_path.with_suffix(".h"))]}
            return TFLiteModelHandler(output_model_path, model_attributes=additional_files)

        return TFLiteModelHandler(output_model_path)
