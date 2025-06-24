# Qwen2.5-0.5B
This folder contains a sample use case of Olive to optimize and quantize [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) model using existing tools.

## Quantization Workflows
- CPU: *PyTorch Model -> SpinQuant -> Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model -> Intel® Neural Compressor Dynamic 8 bit lm_head layer*

## Config file 
The workflow in Config file: [qwen2_5-0.5B_Spinquant_RTN_INC_4bit.json](qwen2_5-0.5B_Spinquant_RTN_INC_4bit.json) executes the above workflow.

## Installation
```bash
python3.10 -m venv olive_env
source olive_env/bin/activate
cd <Olive_directory_path>
pip install -e .
cd examples/Qwen2_5/nxp/
pip install -r requirements.txt
```
**NOTE:**

- For access to the [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) model you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

## Run Optimization
The optimization techniques to run are specified in the relevant config json file.

Run using Olive interface:
```bash
olive run --config qwen2_5-0.5B_Spinquant_RTN_INC_4bit.json
```

or run simply with python code:

```bash
from olive.workflows import run as olive_run
olive_run("qwen2_5-0.5B_Spinquant_RTN_INC_4bit.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
