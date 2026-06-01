# include "DenseLayer.hpp"
# include "Tensor.hpp"
#include <random>
#include <cmath>

// 1. Constructor with Random Xavier/Glorot Initialization
DenseLayer::DenseLayer(int input_dim, int output_dim) 
    : weights(input_dim, output_dim), biases(1, output_dim) {
    
    // Calculate Xavier/Glorot standard deviation and initializing weights
    float stddev = std::sqrt(2.0f / (input_dim + output_dim));
    weights.randomize_normal(0.0f, stddev);
    
}

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
    weight_gradient = (input_cache.transpose()).multiply(output_gradient);

    // Calculate bias_gradient by summing columns across all batch rows
    bias_gradient = Tensor(1, biases.cols); 
    for (int r = 0; r < output_gradient.rows; ++r) {
        for (int c = 0; c < output_gradient.cols; ++c) {
            bias_gradient(0, c) += output_gradient(r, c);
        }
    }

    return input_gradient;
}

std::vector<Tensor*> DenseLayer::get_parameters(){
    return {&weights,&biases};
}

std::vector<Tensor*> DenseLayer::get_gradients(){
    return {&weight_gradient,&bias_gradient};
}