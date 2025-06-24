# H2o-danube3-500m-chat
This folder contains a sample use case of Olive to optimize and quantize [h2oai/h2o-danube3-500m-chat](https://huggingface.co/h2oai/h2o-danube3-500m-chat) model using existing tools.

## Quantization Workflows
- CPU: *PyTorch Model -> SpinQuant -> Onnx Model -> Intel® Neural Compressor 4 bit Quantized Onnx Model -> Intel® Neural Compressor Dynamic 8 bit lm_head layer*

## Config file 
The workflow in Config file: [danube3_Spinquant_RTN_INC_4bit.json]executes the above workflow.

## Installation
```bash
python3.10 -m venv olive_env
source olive_env/bin/activate
cd <Olive_directory_path>
pip install -e .
cd examples/danube3/nxp/
pip install -r requirements.txt
```
**NOTE:**

- For access to the [h2oai/h2o-danube3-500m-chat](https://huggingface.co/h2oai/h2o-danube3-500m-chat) model you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

## Run Optimization
CPU:
```bash
python danube.py --optimize --config danube3_Spinquant_RTN_INC_4bit.json

or run using Olive interface

olive run --config danube3_Spinquant_RTN_INC_4bit.json
```

**NOTE:** You can run the optimization for a locally saved model by setting the `--model_id` to the path of the model.

## Test Inference
To test inference on the model run the script with `--inference`
```bash
python danube.py --config danube3_Spinquant_RTN_INC_4bit.json --inference
```

**NOTE:**
- You can provide you own prompts using `--prompt` argument. For example:
```bash
python danube.py --config danube3_Spinquant_RTN_INC_4bit.json --inference --prompt "What is the capital of France?"
```
- `--max_length` can be used to specify the maximum length of the generated sequence.
