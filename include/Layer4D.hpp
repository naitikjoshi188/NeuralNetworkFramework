#pragma once

#include <vector>
#include "Tensor4D.hpp"

class Layer4D {
public:
    virtual ~Layer4D() = default;

    virtual Tensor4D forward(const Tensor4D& input) = 0;
    virtual Tensor4D backward(const Tensor4D& output_gradient) = 0;
};

class Network4D {
private:
    std::vector<Layer4D*> layers;

public:
    Network4D() = default;
    
    ~Network4D() {
        for (auto &it : layers) {
            delete it;
            it = nullptr;
        }
    }

    void add(Layer4D* layer) {
        layers.push_back(layer);
    }

    Tensor4D forward(Tensor4D input) {
        for (auto* layer : layers) {
            input = layer->forward(input);
        }
        return input;
    }

    Tensor4D backward(Tensor4D output_gradient) {

        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            output_gradient = (*it)->backward(output_gradient);
        }
        return output_gradient;
    }
};