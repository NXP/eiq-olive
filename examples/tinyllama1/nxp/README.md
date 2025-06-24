# TinyLlama-1.1B-Chat-v1.0
This folder contains a sample use case of Olive to optimize and quantize [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model using existing tools.

## Quantization Workflows
- CPU: *PyTorch Model -> SpinQuant -> Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model -> Intel® Neural Compressor Dynamic 8 bit lm_head layer*

## Config file 
The workflow in Config file: [tinyllama1_1B_Spinquant_RTN_INC_4bits.json](tinyllama1_1B_Spinquant_RTN_INC_4bits.json) executes the above workflow.

## Installation
```bash
python3.10 -m venv olive_env
source olive_env/bin/activate
cd <Olive_directory_path>
pip install -e .
cd examples/tinyllama1/nxp/
pip install -r requirements.txt
```
**NOTE:**

- For access to the [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

## Run Optimization
The optimization techniques to run are specified in the relevant config json file.

Run using Olive interface:
```bash
olive run --config tinyllama1_1B_Spinquant_RTN_INC_4bits.json
```

or run simply with python code:

```bash
from olive.workflows import run as olive_run
olive_run("tinyllama1_1B_Spinquant_RTN_INC_4bits.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
