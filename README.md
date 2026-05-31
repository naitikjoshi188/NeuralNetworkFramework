# Custom C++ Neural Network Framework

A modular, object-oriented deep learning framework implemented completely from scratch in C++ without external linear algebra libraries. 

## Features
* **Custom Tensor Engine:** Optimized 2D tensor operations featuring cache-friendly `IKJ` matrix multiplication routing.
* **Memory Efficient:** Native use of constructor initialization lists and constant reference passing to minimize CPU overhead.
* **Strict Memory & Shape Safety:** Bound checks via automated assertions to guarantee algebraic validity at runtime.

## Project Structure
* `include/` — Header files (`Tensor.hpp`, `Layer.hpp`)
* `src/` — Implementation logic (`Tensor.cpp`)