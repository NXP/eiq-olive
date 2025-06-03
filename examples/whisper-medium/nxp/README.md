# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Whisper-medium](https://huggingface.co/openai/whisper-medium) model using ONNXRuntime tools.

## Quantization Workflows
- CPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> OnnxMatMul4Quantizer ->Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*

Outputs the final model and latency results.

## Prerequisites
### Clone the repository and install Olive

If you want to run the optimization pipeline with Intel® Neural Compressor, please make sure that `olive-ai[inc]` is installed.

### Pip requirements
Install the necessary python packages:
```
pip install -r requirements.txt
```

Note: Multilingual support requires onnxruntime>=1.16.0

## Run the config to optimize the model
First, install required packages according to passes.
```bash
olive run --config whisper_medium_ONNX_4bits.json --setup
```

Then, optimize the model

On Linux:
```bash
olive run --config whisper_medium_ONNX_4bits.json 2> /dev/null
```

## Test the transcription of the optimized model
```bash
python test_transcription.py --config whisper_{device}_{precision}.json [--audio_path AUDIO_PATH] [--language LANGUAGE] [--task {transcribe,translate}] [--predict_timestamps]

# For example, to test CPU, INT8 with default audio path
python test_transcription.py --config whisper_cpu_int8.json
```

- `--audio_path` Optional. Path to audio file. If not provided, will use a default audio file.

- `--language` Optional. Language spoken in audio. Default is `english`. Only used when `--multilingual` is provided to `prepare_whisper_configs.py`

- `--task` Optional. Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate'). Default is `transcribe`. Only used
when `--multilingual` is provided to `prepare_whisper_configs.py`

- `--predict_timestamps` Optional. Whether to predict timestamps with the text. Default is `False`. Only used when `--enable_timestamps` is provided to `prepare_whisper_configs.py`

## FAQ
The following are some common issues that may be encountered when running this example.

1. Whenever you install a new version of onnxruntime (such as ort-nightly), you may need to delete the `cache` folder and run the workflow again. This is because the cache doesn't
distinguish between different versions of onnxruntime and will use the cached models from a previous run. There might be incompatibilities between the cached models and the new
version of onnxruntime.

2. If you run out of space in your temp directory, you can add `--tempdir .` to the workflow command to use the current directory as the temp directory root. `.` can be replaced with any other directory with sufficient disk space and write permission.
