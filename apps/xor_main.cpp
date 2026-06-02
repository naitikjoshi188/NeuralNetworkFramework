#include <iostream>
#include <vector>
#include <cmath>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"

int main(){
    //1. setting the XOR sheet
    Tensor X(4,2);
    X.data = {0.0f , 0.0f ,
              1.0f , 0.0f ,
              0.0f , 1.0f ,
              1.0f , 1.0f};
    
    Tensor Y_true(4,1);
    Y_true.data = {0.0f ,
                   1.0f ,
                   1.0f ,
                   0.0f};
    
    //2. Building a Multi-layer network to solve XOR
    Network net;

    //Hidden layer: 2 inputs -> 8 hidden neurons
    net.add(new DenseLayer(2,8));
    //non-linear activation layer
    net.add(new ReLULayer());
    //Hidden Layer: 8 hidden neurons -> 1 prediciton
    net.add(new DenseLayer(8,1));

    //3. Initialize Loss and Optimizer
    MSELoss loss_fn;
    SGDOptimizer optimizer(0.1f);

    std::cout << "========================================\n";
    std::cout << "Starting custom engine training on XOR...\n";
    std::cout << "========================================\n";

    // 4. The Core Training Loop
    for (int epoch = 1; epoch <= 2000; ++epoch) {
        // Step A: Forward Pass
        Tensor Y_pred = net.forward(X);

        // Step B: Compute Loss Cost
        float loss = loss_fn.forward(Y_pred, Y_true);

        // Step C: Backward Pass (Backpropagation)
        Tensor loss_grad = loss_fn.backward(Y_pred, Y_true);
        net.backward(loss_grad);

        // Step D: Update Parameters (SGD)
        optimizer.step(net);

        // Print progress every 50 epochs to watch the error drop
        if (epoch % 200 == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << " | MSE Loss: " << loss << "\n";
        }
    }

    // 5. Verify the Final Predictions
    std::cout << "\n========================================\n";
    std::cout << "Training complete! Final Predictions:\n";
    std::cout << "========================================\n";
    
    Tensor X_test(4, 2);
    X_test.data = {
        0.05f, 0.05f,  // Close to (0, 0) -> should be 0
        0.92f, 0.08f,  // Close to (1, 0) -> should be 1
        0.03f, 0.95f,  // Close to (0, 1) -> should be 1
        0.91f, 0.89f   // Close to (1, 1) -> should be 0
    };

    // Correct mathematical XOR targets for these noisy coordinates
    std::vector<float> Y_test_targets = {0.0f, 1.0f, 1.0f, 0.0f};

    Tensor test_pred = net.forward(X_test);
    for (int i = 0; i < 4; ++i) {
        std::cout << "Input: (" << X_test.data[i * 2] << ", " << X_test.data[i * 2 + 1] 
                  << ") -> Target: " << Y_test_targets[i] 
                  << " | Engine Output: " << test_pred.data[i] << "\n";
    }

    return 0;
}