#pragma once

# include <vector>
# include "Layer.hpp"
# include "Tensor.hpp"

class DenseLayer : public Layer{
private:
    //declaring weight and bias tensors for a layer as well as input tensor as it will be required for backward pass

    Tensor weights;
    Tensor biases;
    Tensor input_cache;
    Tensor weight_gradient;
    Tensor bias_gradient;

public:
    //defining constructor and overloading forward and backword functions

    DenseLayer(int input_dim,int output_dim);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

    std::vector<Tensor*> get_parameters() override;
    std::vector<Tensor*> get_gradients() override;
};