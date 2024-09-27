# Hardware Exploration Framework

This repository is part of the deliverable by **[EMS RPTU](https://ems.eit.uni-kl.de/start)** for the [SustainML](https://sustainml.eu/index.php) project. It includes an initial version of a toolchain designed to convert **[Brevitas](https://github.com/Xilinx/brevitas)-quantized ONNX models** into **High-Level Synthesis (HLS)** code for FPGA-based implementations. While this is an early version and not fully complete, it is functional and capable of generating code for various versions of the **UNet model**, with successful compilation and synthesis of the resulting HLS code.

## Repository Overview

This repository contains two primary folders:

1. **Generator Folder**:
    - This folder contains a **toolkit** for converting ONNX models into HLS code.
    - It accepts ONNX models that have been quantized using **Brevitas** and produces C++ files and header files for FPGA synthesis.
    - The toolkit provides various transformation and optimization features, such as removing unsupported nodes, adding split nodes, and setting FIFO depth.
    - For detailed instructions on how to use the toolkit, please refer to the **detailed README** provided inside this folder.

2. **Hardware Library Folder**:
    - This folder contains a **modified version of the FINN HLS library**.
    - The library has been updated to add support for various variations of convolution layers and activation functions.
    - A more **abstract README** is provided within the folder, summarizing the modifications and usage of the library.

## Key Features

- **ONNX to HLS Conversion**: Supports converting Brevitas-quantized ONNX models to HLS code.
- **UNet Model Support**: Successfully generates HLS code for different versions of the UNet model.
- **Code Compilation and Synthesis**: The generated code compiles and synthesizes successfully on FPGA platforms.
- **Modifications to FINN HLS Library**: Adds support for multiple variations of convolution layers and activation functions.

## Current Limitations

This is an **initial version** of the toolkit and library and is **not fully complete**. However, the existing functionality is already capable of producing usable HLS code for certain types of models, particularly the UNet model. Future updates will aim to improve the toolkit's capabilities, supporting more complex models and optimizations.

## Future Work

- Improve support for additional ONNX layers and architectures.
- Extend compatibility for models beyond the UNet family.
- Add more robust error handling and diagnostic features.
- Refine and optimize hardware performance further.

## Getting Started

For instructions on how to use the toolkit and library, please refer to the **[README](Generator/README.md)** files inside the **Generator** folder.
