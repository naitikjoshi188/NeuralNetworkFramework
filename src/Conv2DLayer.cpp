# include "Conv2DLayer.hpp"

Conv2DLayer::Conv2DLayer(size_t in_c,size_t out_c,size_t k_size,size_t s,size_t p)
:   in_channels(in_c),
    out_channels(out_c),
    kernel_size(k_size),
    stride(s),
    padding(p),
    //constructing 4D tensors
    weights(out_c, in_c,k_size,k_size),
    d_weights(out_c, in_c,k_size,k_size),
    //sizing bias vectors
    biases(out_c, 0.0f),
    d_biases(out_c, 0.0f)
{}

Tensor4D Conv2DLayer::forward(const Tensor4D& input){
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

Tensor4D Conv2DLayer::backward(const Tensor4D& d_output,const Tensor4D& input){
    size_t N = input.N,H=input.H,W=input.W,C=input.C;
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

                                d_weights(j, c_in, kh, kw) += d_output(i, j, l, m) * input(i, c_in, l + kh, m + kw);
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