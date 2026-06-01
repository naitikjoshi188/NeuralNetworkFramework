#pragma once

# include "Layer.hpp"
# include <vector>

class Optimizer{
protected:
    float lr; // learning rate

public:
    Optimizer(float learning_rate) : lr(learning_rate){};
    virtual ~Optimizer() = default;

    // Pure virtual function that each specific optimizer must implement
    virtual void step(Network& net) = 0;
};

class SGDOptimizer : public Optimizer {
public:
    // Constructor passes the learning rate directly to base Optimizer constructor
    SGDOptimizer(float learning_rate) : Optimizer(learning_rate) {}

    // Overriding the step function to implement the SGD loop
    void step(Network& net) override;
};