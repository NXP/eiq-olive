# Example usage of eIQ Olive passes
## Introduction
This short example shows how to use the eIQ Olive passes added to the native Olive program.

To avoid problems, execute the commands in [Example workflow](#example-workflow) in the correct numerical order.

## Example workflow

1. Install necessary prerequisites and download the example model by executing:

    ```
    pip install -r requirements.txt
    python download_model.py
    ```

2. Run an example pass that converts the `ONNX` example model to `TFLite`:
    
    ```
    olive run --config onnx2tflite.json
    ```

    The results will be stored in `outputs-tflite-convert/`.

3. Run an example pass that quantizes the `ONNX` example model:

    ```
    olive run --config onnx2quant.json
    ```

    The results will be stored in `outputs-quant/`.

4. Run an example pass that converts the `TFLite` model (created in step 2.) to `Neutron` accelerated model:

    ```
    olive run --config neutron_conversion.json
    ```
    
    The results will be stored in `outputs-neutron-convert/`.

5. Run an example pass that converts the `TFLite` model (created in step 2.) to `Vela`:

    ```
    olive run --config vela_conversion.json
    ```

    The results will be stored in `outputs-vela-convert/`.
