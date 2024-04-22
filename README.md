# ShanFrame

ShanFrame is a **TinyAI model compiler** for **MCU devices**. It takes a tflite model as input, and generates optimized C code, so that the peak sram usage and latency can be minimized. 

## Overview
<p align="center">
    <img src="./assets/figures/Architecture.drawio.png" alt="architecture diagram" width="540">
</p>

According to the input model structure, ShanFrame utilizes the inter-layer connectivity information to apply optimizations. 

ShanFrame applies the following unique optimizations compared to other AI compilers:
- **Input-output overlapping**: ShanFrame tries to overlap an OP's input and output in memory space as much as possible. With overlapped input and output tensors, the fragmentation level of the model can be reduced and the sram usage for each op is minimized, reducing the peak memory.
- **Data layout fit**: ShanFrame determines the data layout of each intermediate tensor so that the operators can be conducted with highest efficiency.
- **Output tensor pre-padding**: In each OP, if the sram size allows (does not exceed the set sram target), ShanFrame will pre-pad an OP's output tensor instead of doing padding in the next fused OP to reduce latency. 

## Code Structure

`shan_frame` contains the main package of ShanFrame to compile TFLite model into optimized C code.

`example` contains example TFLite models for evaluation and demonstration.

`assets` contains misc assets.

`.vscode` contains dev IDE settings.

## Requirement

- Python 3.11+

## Setup and Run

TBA

## Coding Style

This project follows [Python Code Style Guide](https://peps.python.org/pep-0008/), with following notes. These ensures the code is easily understandable for maintainence. 

- [Type hints](https://docs.python.org/3/library/typing.html): Use as much as type hints so that every variable has known type. 
- Type Checking: If using [Visual Studio Code](https://code.visualstudio.com/) as IDE, use [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) extension and make sure the new code passes `typeCheckingMode: standard`. 

## References

[2D bin packing for storage allocation](http://adambuchsbaum.com/papers/dsa-stoc03.pdf)

[2D bin packing application in memory scheduling](https://arxiv.org/pdf/2305.01497.pdf)