# Qwen/Qwen3-1.7B
This folder contains a sample use case of Olive to optimize and quantize [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) model using existing tools.

## Quantization Workflows
- CPU: *PyTorch Model -> SpinQuant -> Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model -> Intel® Neural Compressor Dynamic 8 bit lm_head layer*

## Config file 
The workflow in Config file: [qwen3-1.7B_Spinquant_INC_RTN_4bit.json](qwen3-1.7B_Spinquant_INC_RTN_4bit.json) executes the above workflow.

## Installation
```bash
python3.10 -m venv olive_env
source olive_env/bin/activate
cd Olive
pip install -e .
cd examples/Qwen3/nxp/Qwen3-1.7B
pip install -r requirements.txt
```
**NOTE:**
Qwen3 architecture is not supported in last official onnxruntime-genai version (0.8.1). In this case a pre-release (nightly build) version is required.
Steps to install: 

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
git checkout 5f7fc49171fac1e5539401ad
pip install cmake ninja cmake
```
Note - CMake 3.26 or higher is required.

```bash
pip install wheel
python build.py
pip install build/Linux/Release/wheel/onnxruntime_genai-0.9.0.dev0-cp310-cp310-linux_x86_64.whl
```

**NOTE:**

- Access to the [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) model is gated and therefore you will need to request access to the model. Once you have access to the model, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

## Run Optimization
The optimization techniques to run are specified in the relevant config json file.

Run using Olive interface:
```bash
olive run --config qwen3-1.7B_Spinquant_INC_RTN_4bit.json
```

or run simply with python code:

```bash
from olive.workflows import run as olive_run
olive_run("qwen3-1.7B_Spinquant_INC_RTN_4bit.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

## FAQ
The following are some common issues that may be encountered when running this example.

1. Whenever you install a new version of onnxruntime (such as ort-nightly), you may need to delete the `cache` folder and run the workflow again. This is because the cache doesn't
distinguish between different versions of onnxruntime and will use the cached models from a previous run. There might be incompatibilities between the cached models and the new
version of onnxruntime.

2. If you run out of space in your temp directory, you can add `--tempdir .` to the workflow command to use the current directory as the temp directory root. `.` can be replaced with any other directory with sufficient disk space and write permission.