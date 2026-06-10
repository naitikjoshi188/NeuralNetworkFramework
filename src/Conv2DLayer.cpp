# include "Conv2DLayer.hpp"
#include <random>
#include <cmath>

Conv2DLayer::Conv2DLayer(size_t in_c, size_t out_c, size_t k_size, size_t s, size_t p)
:   in_channels(in_c),
    out_channels(out_c),
    kernel_size(k_size),
    stride(s),
    padding(p),
    // Constructing 4D tensors (initially default or zero-allocated)
    weights(out_c, in_c, k_size, k_size),
    d_weights(out_c, in_c, k_size, k_size),
    // Sizing bias vectors
    biases(out_c, 0.0f),
    d_biases(out_c, 0.0f),
    input_cache(0, 0, 0, 0)
{
    // 1. Calculate fan_in for He Initialization 
    // fan_in = input_channels * kernel_height * kernel_width
    float fan_in = static_cast<float>(in_c * k_size * k_size);
    float std_dev = std::sqrt(2.0f / fan_in);

    // 2. Setup standard normal distribution random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);

    // 3. Populate the weights Tensor4D data vector with random values
    // Accessing your underlying weight data vector directly
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = dist(gen);
    }
}


Tensor4D Conv2DLayer::forward(const Tensor4D& input){
    input_cache = input;
    
    size_t N = input.N,H=input.H,W=input.W;
    size_t H_out = H - kernel_size +1;
    size_t W_out = W - kernel_size +1;

    Tensor4D output(N,out_channels,H_out,W_out);

    for (size_t i=0;i<N;i++){
        for (size_t j=0;j<out_channels;j++){
            for (size_t l=0;l<H_out;l++){
                for (size_t m=0;m<W_out;m++){
                    float sum = 0.0f; 

                    for (size_t c_in = 0; c_in < in_channels; c_in++) {
                        for (size_t kh = 0; kh < kernel_size; kh++) {
                            for (size_t kw = 0; kw < kernel_size; kw++) {
                                sum += input(i, c_in, l + kh, m + kw) * weights(j, c_in, kh, kw);
                            }
                        }
                    }

                    output(i, j, l, m) = sum + biases[j];
                }
            }
        }
    }
    return output;
}

Tensor4D Conv2DLayer::backward(const Tensor4D& d_output){
    size_t N = input_cache.N,H=input_cache.H,W=input_cache.W,C=input_cache.C;
    size_t H_out = H - kernel_size +1;
    size_t W_out = W - kernel_size +1;

    Tensor4D d_input(N,in_channels,H,W);
    d_weights.fill_zero();
    std::fill(d_biases.begin(),d_biases.end(),0.0f);

    for (size_t i=0;i<N;i++){
        for (size_t j=0;j<out_channels;j++){
            for (size_t l=0;l<H_out;l++){
                for (size_t m=0;m<W_out;m++){
                    for (size_t c_in=0;c_in<in_channels;c_in++){
                        for (size_t kh = 0; kh < kernel_size; kh++) {
                            for (size_t kw = 0; kw < kernel_size; kw++) {
                                if (c_in == 0 && kh == 0 && kw == 0) 
                                {d_biases[j] += d_output(i, j, l, m);}

                                d_weights(j, c_in, kh, kw) += d_output(i, j, l, m) * input_cache(i, c_in, l + kh, m + kw);
                                d_input(i, c_in, l + kh, m + kw) += d_output(i, j, l, m) * weights(j, c_in, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }
    return d_input;
}

void Conv2DLayer::update_parameters(float learning_rate){
    for (size_t o=0;o<out_channels;o++){
        for (size_t i=0;i<in_channels;i++){
            for (size_t kh=0;kh<kernel_size;kh++){
                for (size_t kw=0;kw<kernel_size;kw++){
                    weights(o,i,kh,kw) -= learning_rate*d_weights(o,i,kh,kw);
                }
            }
        }
    }

    for (size_t o=0;o<out_channels;o++){
        biases[o] -= learning_rate*d_biases[o];
    }
}