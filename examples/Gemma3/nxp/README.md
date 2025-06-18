# gemma-3-1b-it-qat-int4-unquantized
This folder contains a sample use case of Olive to optimize and quantize [google/gemma-3-1b-it-qat-int4-unquantized](https://huggingface.co/google/gemma-3-1b-it-qat-int4-unquantized) model using existing tools.

## Quantization Workflows
- CPU: *PyTorch Model -> Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model -> Intel® Neural Compressor Dynamic 8 bit lm_head layer*

## Config file 
The workflow in Config file: [gemma3_1B_RTN_INC_4bits.json](gemma3_1B_RTN_INC_4bits.json) executes the above workflow.

## Installation
```bash
python3.10 -m venv olive_env
source olive_env/bin/activate
cd Olive
pip install -e .
cd examples/Gemma3/nxp/
pip install -r requirements.txt
```
**NOTE:**

- Access to the [google/gemma-3-1b-it-qat-int4-unquantized](https://huggingface.co/google/gemma-3-1b-it-qat-int4-unquantized) model is gated and therefore you will need to request access to the model. Once you have access to the model, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

## Run Optimization
The optimization techniques to run are specified in the relevant config json file.

Run using Olive interface:
```bash
olive run --config gemma3_1B_RTN_INC_4bits.json
```

or run simply with python code:

```bash
from olive.workflows import run as olive_run
olive_run("gemma3_1B_RTN_INC_4bits.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
