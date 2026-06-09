#pragma once

#include "Layer.hpp"
#include "Tensor.hpp"
#include "Tensor4D.hpp"
#include <vector>

class ReLULayer: public Layer{
private:
    Tensor input_cache;

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
};

class SoftmaxLayer: public Layer{
private:
    Tensor cached_output;

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
};

class ReLU4DLayer{
private:
    Tensor4D input_cache;

public:
    Tensor4D forward(const Tensor4D& input);
    Tensor4D backward(const Tensor4D& d_output);
};