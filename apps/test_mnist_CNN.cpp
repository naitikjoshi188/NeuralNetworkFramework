#include <iostream>
#include <vector>
#include <cmath>
#include "Utils.hpp"
#include "Layer4D.hpp"
#include "Layer.hpp"
#include "Conv2DLayer.hpp"
#include "ActivationLayer.hpp"
#include "MaxPool2D.hpp"
#include "FlattenLayer.hpp"
#include "DenseLayer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

int main() {
    // 1. Load and Preprocess Datasets
    std::cout << "=== Initializing Data Engine ===\n";
    Tensor X_train, Y_train;
    Tensor X_test, Y_test;
    
    std::cout << "Loading MNIST Training Set...\n";
    DataUtils::load_csv("data/mnist_train.csv", X_train, Y_train, 0); 
    std::cout << "Loaded " << X_train.rows << " training samples.\n";

    std::cout << "Loading MNIST Testing Set...\n";
    DataUtils::load_csv("data/mnist_test.csv", X_test, Y_test, 0); 
    std::cout << "Loaded " << X_test.rows << " testing samples.\n\n";

    std::cout << "Executing feature scaling normalization...\n";
    DataUtils::normalize_features(X_train);
    DataUtils::normalize_features(X_test);
    std::cout << "Data preprocessing complete!\n\n";

    // 2. Initialize Core Network Pipelines (16 Filters Configured)
    std::cout << "=== Building CNN Architecture ===\n";
    Conv2DLayer* conv1 = new Conv2DLayer(1, 16, 3); 
    
    Network4D feature_extractor;
    feature_extractor.add(conv1);
    feature_extractor.add(new ReLU4DLayer());
    feature_extractor.add(new MaxPool2D(2, 2));

    FlattenLayer flatten;

    Network classifier;
    DenseLayer* dense1 = new DenseLayer(2704, 64);
    ReLULayer* relu_classifier = new ReLULayer();
    DenseLayer* dense2 = new DenseLayer(64, 10);   

    classifier.add(dense1);
    classifier.add(relu_classifier);
    classifier.add(dense2);
    classifier.add(new SoftmaxLayer());
    std::cout << "Pipeline configurations successfully locked.\n\n";

    AdamOptimizer optimizer(0.003f);
    CrossEntropyLoss loss_fn;
    float learning_rate = 0.003f;

    // 3. Hyperparameters
    size_t batch_size = 32;
    size_t num_train_batches = X_train.rows / batch_size;
    size_t num_test_batches = X_test.rows / batch_size;
    size_t epochs = 10; // Set to 10 epochs

    std::cout << "=== Starting Dual Training & Evaluation Loop ===\n";
    std::cout << "Training Batches: " << num_train_batches 
              << " | Testing Batches: " << num_test_batches << "\n\n";

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        // ==========================================
        // PHASE A: TRAINING BLOCK
        // ==========================================
        if (epoch > 0 && epoch % 2 == 0) {
            learning_rate *= 0.5f;
            
            // Update the learning rate inside your 2D Adam optimizer structure
            optimizer.lr = (learning_rate); 
            
            std::cout << ">> Learning rate decayed. New Alpha: " << learning_rate << "\n";
        }
        float train_epoch_loss = 0.0f;
        size_t train_correct = 0;

        for (size_t b = 0; b < num_train_batches; b++) {
            Tensor4D x_batch = DataUtils::get_spatial_batch(X_train, b, batch_size, 1, 28, 28);
            Tensor y_raw = DataUtils::get_batch(Y_train, b, batch_size);
            Tensor y_targets = DataUtils::one_hot_encoder(y_raw, 10);

            // Forward
            Tensor4D spatial_features = feature_extractor.forward(x_batch);
            Tensor flat_features = flatten.forward(spatial_features);
            Tensor probabilities = classifier.forward(flat_features);

            // Loss & Metrics Accumulation
            train_epoch_loss += loss_fn.forward(probabilities, y_targets) * batch_size;
            for (size_t i = 0; i < batch_size; i++) {
                int target_class = static_cast<int>(y_raw(i, 0));
                int predicted_class = 0;
                float max_prob = -1.0f;
                for (int c = 0; c < 10; c++) {
                    if (probabilities(i, c) > max_prob) {
                        max_prob = probabilities(i, c);
                        predicted_class = c;
                    }
                }
                if (predicted_class == target_class) { train_correct++; }
            }

            // Backward & Step
            Tensor d_loss = loss_fn.backward(probabilities, y_targets);
            Tensor d_classifier = classifier.backward(d_loss);
            Tensor4D d_spatial_bridge = flatten.backward(d_classifier);
            feature_extractor.backward(d_spatial_bridge);

            optimizer.step(classifier);
            conv1->update_parameters(learning_rate);
        }

        float avg_train_loss = train_epoch_loss / static_cast<float>(X_train.rows);
        float train_accuracy = (static_cast<float>(train_correct) / static_cast<float>(X_train.rows)) * 100.0f;

        // ==========================================
        // PHASE B: INDEPENDENT TESTING BLOCK
        // ==========================================
        size_t test_correct = 0;

        for (size_t b = 0; b < num_test_batches; b++) {
            // Stream validation batches through the exactly same tools
            Tensor4D x_test_batch = DataUtils::get_spatial_batch(X_test, b, batch_size, 1, 28, 28);
            Tensor y_test_raw = DataUtils::get_batch(Y_test, b, batch_size);

            // Forward evaluation pass only (No backward pass, no optimizer tracking)
            Tensor4D test_spatial = feature_extractor.forward(x_test_batch);
            Tensor test_flat = flatten.forward(test_spatial);
            Tensor test_probabilities = classifier.forward(test_flat);

            // Track generalization accuracy
            for (size_t i = 0; i < batch_size; i++) {
                int target_class = static_cast<int>(y_test_raw(i, 0));
                int predicted_class = 0;
                float max_prob = -1.0f;
                for (int c = 0; c < 10; c++) {
                    if (test_probabilities(i, c) > max_prob) {
                        max_prob = test_probabilities(i, c);
                        predicted_class = c;
                    }
                }
                if (predicted_class == target_class) { test_correct++; }
            }
        }

        float test_accuracy = (static_cast<float>(test_correct) / static_cast<float>(num_test_batches * batch_size)) * 100.0f;

        // Log results side-by-side
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << " | Train Loss: " << avg_train_loss 
                  << " | Train Acc: " << train_accuracy << "%"
                  << " | Test Acc: " << test_accuracy << "%\n";
    }

    std::cout << "\nTraining and verification completed completely!\n";
    return 0;
}