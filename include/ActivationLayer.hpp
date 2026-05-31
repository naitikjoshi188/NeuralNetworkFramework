#pragma once

#include "Layer.hpp"
#include "Tensor.hpp"
#include <vector>

class ReLULayer: public Layer{
private:
    Tensor input_cache;

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
};