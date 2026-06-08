# pragma once

# include <Tensor4D.hpp>
# include <vector>

class Conv2DLayer{
public:
    //Hyperparameters
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size; //considering only square kernels
    size_t stride;
    size_t padding;

    //learnable parameters
    Tensor4D weights;          //shape: (out_c,in_c,kernel_s,kernel_s)
    std::vector<float> biases; //shape: (out_channels)

    //gradients for optimization
    Tensor4D d_weights;
    std::vector<float> d_biases;

    //Constructor
    Conv2DLayer(size_t in_c,size_t out_c,size_t k_size,size_t s=1,size_t p=0);

    //Core execution phases
    Tensor4D forward(const Tensor4D& input);
    Tensor4D backward(const Tensor4D& d_output,const Tensor4D& input);

    //parameter updates
    void update_parameters(float learning_rate);
};