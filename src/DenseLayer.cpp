# include "DenseLayer.hpp"
# include "Tensor.hpp"

//defining forward and backward passes

Tensor DenseLayer::forward(const Tensor& input){
    input_cache = input;
    Tensor result = input.multiply(weights);
    for (int r = 0; r < result.rows; ++r) {
        for (int c = 0; c < result.cols; ++c) {
            result(r, c) += biases(0, c);
        }
    }
    return result;
}

Tensor DenseLayer::backward(const Tensor& output_gradient){
    Tensor input_gradient = output_gradient.multiply(weights.transpose());
    Tensor weight_gradient = (input_cache.transpose()).multiply(output_gradient);
    return input_gradient;
}