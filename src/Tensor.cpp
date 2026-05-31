# include "Tensor.hpp"
# include <iostream>
# include <cassert>

Tensor::Tensor(): rows(0),cols(0){}

Tensor::Tensor(int r,int c): rows(r),cols(c),data(r*c,0.0f){}

Tensor::Tensor(int r,int c,const std::vector<float>& values): rows(r),cols(c),data(values){
    assert(r*c==values.size());
}

Tensor Tensor::transpose()const{
    Tensor result(cols,rows);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            result(j,i) = (*this)(i,j);
        }
    }
    return result;
}

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