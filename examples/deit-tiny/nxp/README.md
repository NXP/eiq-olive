# deit-tiny-patch16-224
This folder contains a sample use case of Olive to optimize and quantize [facebook/deit-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224) model using existing tools.

## Quantization Workflows
- CPU: *PyTorch Model -> Onnx Model -> Onnx Optimized Model -> OnnxQuantization 8 bit RTN Quantized Onnx Model*

## Config file 
The workflow in Config file: [deit-tiny_RTN_ONNX_8bit.json](deit-tiny_RTN_ONNX_8bit.json) executes the above workflow.

## Installation
```bash
python3.10 -m venv olive_env
source olive_env/bin/activate
cd <Olive_directory_path>
pip install -e .
cd examples/deit-tiny/nxp/
pip install -r requirements.txt
```
**NOTE:**

- For access to the [facebook/deit-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224) model you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

## Run Optimization
The optimization techniques to run are specified in the relevant config json file.

Run using Olive interface:
```bash
olive run --config deit-tiny_RTN_ONNX_8bit.json
```

or run simply with python code:

```bash
from olive.workflows import run as olive_run
olive_run("deit-tiny_RTN_ONNX_8bit.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

## FAQ
The following are some common issues that may be encountered when running this example.

1. Whenever you install a new version of onnxruntime (such as ort-nightly), you may need to delete the `cache` folder and run the workflow again. This is because the cache doesn't
distinguish between different versions of onnxruntime and will use the cached models from a previous run. There might be incompatibilities between the cached models and the new
version of onnxruntime.

2. If you run out of space in your temp directory, you can add `--tempdir .` to the workflow command to use the current directory as the temp directory root. `.` can be replaced with any other directory with sufficient disk space and write permission.