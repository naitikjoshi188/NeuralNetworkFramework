# include "Optimizer.hpp"
# include <vector>

void SGDOptimizer::step(Network& net){
    std::vector<Layer*> layers = net.get_layers();
    for (const auto& it:layers){
        std::vector<Tensor*> params = it->get_parameters();
        std::vector<Tensor*> grads = it->get_gradients();
        for (size_t i=0;i<params.size();i++){
            Tensor* p = params[i];
            Tensor* g = grads[i];
            int total_elements = p->rows*p->cols;
            for (int j=0;j<total_elements;j++){
                p->data[j] -= lr * g->data[j];
            }
        }
    }
}