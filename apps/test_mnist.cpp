#include <iostream>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"
#include "Utils.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << "Starting Full MNIST Deep Learning Vision Pipeline\n";
    std::cout << "==================================================\n";

    // 1. Setup Data Preprocessing Space
    Tensor X_train_raw, Y_train_raw;
    Tensor X_test_raw, Y_test_raw;
    
    // Load Training Dataset
    std::cout << "Loading MNIST Training Set..." << std::endl;
    DataUtils::load_csv("data/mnist_train.csv", X_train_raw, Y_train_raw, 0);

    // Load Testing Dataset
    std::cout << "Loading MNIST Testing Set..." << std::endl;
    DataUtils::load_csv("data/mnist_test.csv", X_test_raw, Y_test_raw, 0);

    // Slice down to manageable sizes for fast single-threaded CPU processing
    size_t train_limit = 5000;
    size_t test_limit = 1000;

    Tensor X_train(train_limit, X_train_raw.cols);
    Tensor Y_train(train_limit, 1);
    for(size_t i = 0; i < train_limit; ++i) {
        Y_train(i, 0) = Y_train_raw(i, 0);
        for(size_t j = 0; j < X_train_raw.cols; ++j) {
            X_train(i, j) = X_train_raw(i, j);
        }
    }

    Tensor X_test(test_limit, X_test_raw.cols);
    Tensor Y_test(test_limit, 1);
    for(size_t i = 0; i < test_limit; ++i) {
        Y_test(i, 0) = Y_test_raw(i, 0);
        for(size_t j = 0; j < X_test_raw.cols; ++j) {
            X_test(i, j) = X_test_raw(i, j);
        }
    }

    std::cout << "» Sliced working data to " << X_train.rows << " Train / " << X_test.rows << " Test samples.\n\n";

    // Normalize pixel intensity maps (0-255 -> 0.0-1.0)
    std::cout << "Executing feature scaling normalization..." << std::endl;
    DataUtils::normalize_features(X_train);
    DataUtils::normalize_features(X_test);

    // One-Hot Encode labels for categorical cross-entropy tracking
    Tensor Y_train_hot = DataUtils::one_hot_encoder(Y_train, 10);
    Tensor Y_test_hot  = DataUtils::one_hot_encoder(Y_test, 10);

    // 2. Building the Network Topology
    Network net;
    net.add(new DenseLayer(784, 128));
    net.add(new ReLULayer());
    net.add(new DenseLayer(128, 64));
    net.add(new ReLULayer());
    net.add(new DenseLayer(64, 10));
    net.add(new SoftmaxLayer());

    CrossEntropyLoss loss_fn;
    AdamOptimizer optimizer(0.003f);

    std::cout << "\nArchitecture loaded. Commencing network optimization...\n";

    // 3. The Core Optimization Loop
    int total_epochs = 50; 
    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        
        // Forward propagation across the training space
        Tensor Y_pred = net.forward(X_train);
        float loss = loss_fn.forward(Y_pred, Y_train_hot);

        // Backward propagation & gradient distribution
        Tensor loss_grad = loss_fn.backward(Y_pred, Y_train_hot);
        net.backward(loss_grad);

        // Update framework parameters using Adam
        optimizer.step(net);

        if (epoch % 5 == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << "/" << total_epochs << " | Training Loss: " << loss << "\n";
        }
    }

    // 4. Run Generalization Testing on Unseen Data
    std::cout << "\n==================================================\n";
    std::cout << "Training Concluded! Evaluating on Independent Test Data...\n";
    std::cout << "==================================================\n";
    
    Tensor test_preds = net.forward(X_test);
    int correct_test_predictions = 0;

    for (size_t i = 0; i < X_test.rows; ++i) {
        int true_digit = static_cast<int>(Y_test(i, 0));
        
        // Find argmax to locate the network's top guess
        int predicted_digit = 0;
        float max_prob = test_preds(i, 0);
        for (size_t c = 1; c < test_preds.cols; ++c) {
            if (test_preds(i, c) > max_prob) {
                max_prob = test_preds(i, c);
                predicted_digit = c;
            }
        }

        if (predicted_digit == true_digit) {
            correct_test_predictions++;
        }
    }

    float test_accuracy = (static_cast<float>(correct_test_predictions) / X_test.rows) * 100.0f;
    std::cout << ">> Final Out-of-Sample Test Accuracy: " << test_accuracy << "%\n";

    return 0;
}