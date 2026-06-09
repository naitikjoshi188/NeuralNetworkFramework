# pragma once

# include <Tensor4D.hpp>

class MaxPool2D{
public:
    size_t pool_size;
    size_t stride;

    Tensor4D input_cache;

    MaxPool2D(size_t p_size = 2,size_t s = 2);

    Tensor4D forward(const Tensor4D& input);
    Tensor4D backward(const Tensor4D& d_output);
};