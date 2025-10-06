#
# Copyright 2025 NXP
#
# Licensed under the MIT License.
#
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_tflite_file_path(model_path: str) -> str:
    """Get the path to the TFLite model file.

    If model_path is a file, it is returned as is. If model_path is a directory, it is inferred
    if there is only one .tflite file in the directory, else an error is raised.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    if Path(model_path).is_file():
        return model_path

    tflite_file_names = list(Path(model_path).glob("*.tflite"))
    if len(tflite_file_names) == 1:
        return str(tflite_file_names[0])
    elif len(tflite_file_names) > 1:
        raise ValueError(f"Multiple .tflite model files found in the model folder {model_path}.")
    else:
        raise ValueError(f"No .tflite file found in the model folder {model_path}.")
