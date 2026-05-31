# include <Layer.hpp>

Network::Network(){
}

//forward and backward pass combining all layers

Tensor Network::forward(Tensor input){
    for (const auto& it:layers){
        input = it->forward(input);
    }
    return input;
}

Tensor Network::backward(const Tensor& output_gradient){
    Tensor output = output_gradient;
    for (int i=layers.size()-1;i>=0;i--){
        output = layers[i]->backward(output);
    }
    return output;
}