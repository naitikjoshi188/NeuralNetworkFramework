# include "ActivationLayer.hpp"

Tensor ReLULayer::forward(const Tensor& input){
    input_cache = input;

    Tensor result(input.rows,input.cols);

    for (int i=0;i<input.rows;i++){
        for (int j=0;j<input.cols;j++){
            result(i,j) = (input(i,j)>0)?input(i,j):0.0f;
        }
    }
    return result;
}

Tensor ReLULayer::backward(const Tensor& output_gradient){
    Tensor input_gradient(output_gradient.rows,output_gradient.cols);

    for (int i=0;i<output_gradient.rows;i++){
        for (int j=0;j<output_gradient.cols;j++){
            if (input_cache(i,j)>0){
                input_gradient(i,j)=output_gradient(i,j);
            }
        }
    }

    return input_gradient;
}