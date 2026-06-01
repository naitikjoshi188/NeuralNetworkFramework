# include "Tensor.hpp"
# include <iostream>
# include <cassert>
#include <random>


//defining constructors
Tensor::Tensor(): rows(0),cols(0){}

Tensor::Tensor(int r,int c): rows(r),cols(c),data(r*c,0.0f){}

Tensor::Tensor(int r,int c,const std::vector<float>& values): rows(r),cols(c),data(values){
    assert(r*c==values.size());
}


//defining transpose
Tensor Tensor::transpose()const{
    Tensor result(cols,rows);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            result(j,i) = (*this)(i,j);
        }
    }
    return result;
}


//defining print function
void Tensor::print()const{
    std::cout<<"["<<rows<<"*"<<cols<<"] tensor:\n";
    for (int i=0;i<rows;i++){
        std::cout<<"[";
        for (int j=0;j<cols;j++){
            std::cout<<(*this)(i,j)<<" ";
        }
        std::cout<<"]\n";
    }
    std::cout<<std::endl;
}


//defining multiply function and safegaurding using assert
Tensor Tensor::multiply(const Tensor&B)const{
    assert(cols==B.rows);
    Tensor result(rows,B.cols);
    for (int i=0;i<rows;i++){
        for (int k=0;k<cols;k++){
            float val_A = (*this)(i,k);
            for (int j=0;j<B.cols;j++){
                result(i,j) += val_A*B(k,j);
            }
        }
    }
    return result;
}

//defining random initializer
void Tensor::randomize_normal(float mean, float stddev) {
    // Using a fixed seed (e.g., 1337) ensures the network behaves identically 
    // every time we run it.
    static std::default_random_engine generator(1337);
    std::normal_distribution<float> distribution(mean, stddev);

    for (int i = 0; i < rows * cols; ++i) {
        data[i] = distribution(generator);
    }
}