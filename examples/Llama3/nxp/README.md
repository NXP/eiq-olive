# Llama-3.2-1b
This folder contains a sample use case of Olive to optimize and quantize [meta-llama/Llama-3.2-1b](https://huggingface.co/meta-llama/Llama-3.2-1b) model using existing tools.

## Quantization Workflows
- CPU: *PyTorch Model -> SpinQuant -> Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model -> Intel® Neural Compressor Dynamic 8 bit lm_head layer*

## Config file 
The workflow in Config file: [llama3_2-1B_Spinquant_RTN_INC_4bits.json](llama3_2-1B_Spinquant_RTN_INC_4bits.json) executes the above workflow.

## Installation
```bash
python3.10 -m venv olive_env
source olive_env/bin/activate
cd Olive
pip install -e .
cd examples/Llama3/nxp/
pip install -r requirements.txt
```
**NOTE:**

- Access to the [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model is gated and therefore you will need to request access to the model. Once you have access to the model, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

## Run Optimization
The optimization techniques to run are specified in the relevant config json file.

Run using Olive interface:
```bash
olive run --config llama3_2-1B_Spinquant_RTN_INC_4bits.json
```

or run simply with python code:

```bash
from olive.workflows import run as olive_run
olive_run("llama3_2-1B_Spinquant_RTN_INC_4bits.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
