# pragma once

# include <string>
# include <vector>
# include <fstream>
# include <sstream>
# include <algorithm>
# include "Tensor.hpp"

class DataUtils{
public:
    static void load_csv(const std::string& filepath,Tensor& X_out,Tensor& Y_out,int label_column_idx);

    static void normalize_features(Tensor& X);

    static Tensor one_hot_encoder(const Tensor& Y_true,int num_classes);

    static Tensor get_batch(const Tensor& source, int batch_idx, int batch_size);
};