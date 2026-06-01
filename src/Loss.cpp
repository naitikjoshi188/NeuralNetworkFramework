# include "Loss.hpp"

float MSELoss::forward(const Tensor& pred,const Tensor& true_vals){
    float total_sum = 0.0f;
    int total_elements = pred.rows * pred.cols;

    // Flattening to a single loop maximizes cache efficiency
    for (int i = 0; i < total_elements; i++) {
        float diff = pred.data[i] - true_vals.data[i]; 
        total_sum += diff * diff;
    }

    return total_sum / static_cast<float>(total_elements);
}

Tensor MSELoss::backward(const Tensor& pred,const Tensor& true_vals){
    Tensor gradient(pred.rows,pred.cols);
    int total_elements = pred.rows*pred.cols;

    for (int i=0;i<total_elements;i++){
        gradient.data[i] = (2.0f/static_cast<float>(total_elements))*(pred.data[i]-true_vals.data[i]);
    }
    
    return gradient;
}

