#include <iostream>
#include <chrono>

#include "GLMNET.h"

int main() {
    
    // Setup
    // const int N = 506;
    // const int P = 13;

    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    std::string URL = "/Users/kailashanand/Documents/Developer/MC_OASIS/Project_OASIS/amazing_data_0.csv";

    
    GLMNET glm = GLMNET(X, y, URL); // this initialzes the weights, and loads the data in one go.
    
    glm.train_with_RSLR_on(X, y, 1e4, 1e0, 0.6, 1e-2, 1, false);
    // glm.train_on(X, y, 1e4, 1e-3, 0.1, 0.6, false);

    // std::cout << glm.N << " " << glm.P << std::endl;
    
    std::cout << glm.W << std::endl;

    return 0;
}