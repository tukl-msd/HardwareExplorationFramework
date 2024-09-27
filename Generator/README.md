# ONNX to HLS Generator Toolkit

This project is a **generator toolkit** designed to take an ONNX model quantized using **[Brevitas](https://github.com/Xilinx/brevitas)**, and generate a **High-Level Synthesis (HLS)** representation of it. The generated HLS code is optimized for hardware implementation and uses a library of HLS tools. The toolkit transforms the input ONNX model into an HLS representation in the form of C++ (`.cpp`) and header (`.hpp`) files that are ready for synthesis on FPGA platforms.

## Features

- **ONNX Model Conversion**: Supports conversion of ONNX models, including models quantized using Brevitas and floating-point models.
- **Node Optimizations**: Removes unsupported nodes, merges certain nodes, and adds split nodes for compatibility with HLS tools.
- **Hardware Optimization**: Sets FIFO depths, parallelism, and divides the model into multiple top levels.
- **Generated Output**: The tool generates `.hpp` and `.cpp` files, which include top-level wrappers, profilers, configuration, and parameters for HLS.

## Table of Contents

- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Example](#example)
- [License](#license)
- [Output Files](#output-files)
- [Further Development](#further-development)

## Usage

To use this toolkit, ensure that you have the following dependencies:

### Requirements:

- Python 3.x
- ONNX
- Brevitas (for quantized ONNX models)
- Additional libraries as needed for ONNX manipulation and visualization

You can install the required Python libraries by running:

```bash
pip install onnx brevitas networkx matplotlib
```


To use the toolkit, run the following command in the terminal:

```bash
python onnx_to_hls.py --model <path_to_onnx_model> [additional options]
```

This will take the ONNX model and generate HLS-compatible C++ and header files in the specified output directories.

### Command-Line Arguments

- `--model`: (Required) Path to the ONNX model file.
- `--result_hpp_path`: (Optional) Path to save the generated `.hpp` files (default: `./include_hw/`).
- `--result_cpp_path`: (Optional) Path to save the generated `.cpp` files (default: `./src_hw/`).
- `--input_mem_width`: (Optional) Input memory width (default: 64).
- `--output_mem_width`: (Optional) Output memory width (default: 64).
- `--max_subgraph_weight`: (Optional) Maximum subgraph weight (default: 250,000).
- `--subgraph_weight_margin`: (Optional) Allowed subgraph weight margin (default: 1.1).
- `--brevitas`: (Optional) Flag to indicate if the ONNX model is quantized using Brevitas (default: `False`).
- `--float`: (Optional) Flag to treat the model as a floating-point model (default: `False`).
- `--profiler_json`: (Optional) Path to the profiler JSON file (default: `./profiled_activations.json`).
- `--stop_node`: (Optional) Last node in the graph (default: `''`).

### Example

Here’s an example of how to use the toolkit:

```bash
python onnx_to_hls.py --model ./models/my_model.onnx \
    --result_hpp_path ./include_hw/ \
    --result_cpp_path ./src_hw/ \
    --input_mem_width 64 \
    --output_mem_width 64 \
    --brevitas
```

This command will take the ONNX model located at `./models/my_model.onnx`, and generate `.hpp` files in `./include_hw/` and `.cpp` files in `./src_hw/` based on the model's architecture. The model is assumed to be quantized using Brevitas.

### Output Files

- **Top-Level Wrapper Files**: `dnn_top_level_wrapper.hpp`, `dnn_top_level_wrapper.cpp`
- **Main Logic Files**: `dnn_top_level.cpp`
- **Profilers**: `dnn_profilers.hpp`, `dnn_profilers.cpp`
- **Configuration Files**: `dnn_config.hpp`
- **Parameter Files**: `dnn_params.hpp`
- **Visualization Files**: `my_model_after_onnx.png`, `my_model_after_creating_subgraphs.png` this allows the user to visualize the internal structure of the model after the toolkit has processed it.

### Further Development

- Support for more complex ONNX layers and edge cases will be included in upcoming updates.
- Support for other quantization libraries and formats will be added.
