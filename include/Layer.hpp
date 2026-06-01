#pragma once

# include <vector>
# include "Tensor.hpp"

class Layer{
public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input)=0;

    virtual Tensor backward(const Tensor& output_gradient)=0;

    // Provide default empty parameter arrays for non-parametric layers
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_gradients() { return {}; }
};

class Network{
private:
    std::vector<Layer*> layers;

public:
    //declaring constructors and destructors

    Network();
    ~Network(){
        for (auto &it:layers){
            delete it;
            it = nullptr;
        }
    }

    //defining add,forward and backward pass functions

    void add(Layer* layer){
        layers.push_back(layer);
    }

    Tensor forward(Tensor input);

    Tensor backward(const Tensor& output_gradient);

    //providing access to layer vector
    const std::vector<Layer*>& get_layers() const { return layers; }
};