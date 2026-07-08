#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <sstream>
#include <string>

#include "Eigen/Dense"
#include "Eigen/Core"

// in this project i am using the bias trick - i will augment the bias term into the weight vector, and add a column of ones to the data


float sign(float x) {
    if (x>0) {
        return 1;
    } else if (x<0) {
        return -1;
    }

    return 0;
}

void load_data_to(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, std::string URL) { // this URL will just be some location in local memory

    /*
    NOTE: Conventions that I am assuming:
    1. X is of shape N x (P+1) : so that we can augment with bias column of 1s.
    2. The file:
        a) doesnt have a header
        b) first P columns are the predictor values
        c) the last column contains the response value
    */
    
    std::ifstream data_file;

    const int N = static_cast<int>(X.rows());
    const int P = static_cast<int>(X.cols()) - 1;


    // augment one column of 1s to the data (the first column)
    X.col(0) = Eigen::VectorXf::Ones(N);

    data_file.open(URL);

    if (!(data_file.is_open())) { return; }

    for (int i{0}; i<N; i++) {
        std::string line;
        std::getline(data_file, line);
        std::stringstream ss(line);

        for (int j{1}; j<P+2; j++) {
            
            std::string predictor_val_str;
            std::getline(ss, predictor_val_str, ',');

            if (j == P+1) {
                y(i) = std::stof(predictor_val_str)/1.0;
            } else {
                X(i,j) = std::stof(predictor_val_str)/1.0;
            }

        }

    }



    data_file.close();
}

void write_vector_data_to(std::string URL, std::vector<float> data) {
    std::ofstream data_file;
    data_file.open(URL);

    if (!(data_file.is_open())) { return; }

    for (int i{0}; i<data.size(); i++) {
        data_file << data[i] << ", ";
    }

    data_file.close();
}


float ridge_loss_function(Eigen::Ref<Eigen::VectorXf> W,
                          Eigen::Ref<Eigen::MatrixXf> X,
                          Eigen::Ref<Eigen::VectorXf> y, float reg) {
    
    int N = static_cast<int>(X.rows());

    float data_loss = ((y - X * W).transpose() * (y - X * W)).value();
    float reg_loss = reg * (W.squaredNorm());

    return data_loss/(2*N) + reg_loss/2;


}

float LASSO_loss_function(Eigen::Ref<Eigen::VectorXf> W,
                          Eigen::Ref<Eigen::MatrixXf> X,
                          Eigen::Ref<Eigen::VectorXf> y, float reg) {

    int N = static_cast<int>(X.rows());

    float data_loss = ((y - X * W).transpose() * (y - X * W)).value();
    float reg_loss = reg * (W.lpNorm<1>());

    return data_loss/(2*N) + reg_loss;           

}

std::vector<float> train_weights_with_RIDGE_into(Eigen::Ref<Eigen::VectorXf> W, // of shape (P+1) x 1
                                   Eigen::Ref<Eigen::MatrixXf> X, // of shape N x (P+1)
                                   Eigen::Ref<Eigen::VectorXf> y, // of shape N x 1
                                   float learning_rate, int num_iters, int period, float reg, bool quiet) {

    
    int N = static_cast<int>(X.rows());
    std::vector<float> loss_history;


    // this just performs full on gradient descent
    
    for (int i{0}; i<num_iters; i++) {
        
        Eigen::VectorXf grad_W = (X.transpose() * (X * W - y))/N + reg * W;
        
        W -= learning_rate * grad_W;

        float loss = ridge_loss_function(W, X, y, reg);
        loss_history.push_back(loss);

        if ((i+1) % period == 0 && !(quiet)) {
            std::cout << "Epoch: " << i+1 << " Loss: " << loss << "\n";
        }
    }

    return loss_history;

}

std::vector<float> train_weights_with_LASSO_into(Eigen::Ref<Eigen::VectorXf> W, // of shape (P+1) x 1
                                   Eigen::Ref<Eigen::MatrixXf> X, // of shape N x (P+1)
                                   Eigen::Ref<Eigen::VectorXf> y, // of shape N x 1
                                   float learning_rate, int num_iters, int period, float reg, bool quiet) {

    
    int N = static_cast<int>(X.rows());
    std::vector<float> loss_history;


    // this just performs full on gradient descent
    
    for (int i{0}; i<num_iters; i++) {
        
        Eigen::VectorXf grad_LASSO = W.unaryExpr(&sign);
        Eigen::VectorXf grad_W = (X.transpose() * (X * W - y))/N + reg * grad_LASSO;
        
        W -= learning_rate * grad_W;

        float loss = LASSO_loss_function(W, X, y, reg);
        loss_history.push_back(loss);

        if ((i+1) % period == 0 && !(quiet)) {
            std::cout << "Epoch: " << i+1 << " Loss: " << loss << "\n";
        }
    }

    return loss_history;

}


float find_R_squared_on(Eigen::Ref<Eigen::VectorXf> W, // of shape (P+1) x 1
                        Eigen::Ref<Eigen::MatrixXf> X, // of shape N x (P+1)
                        Eigen::Ref<Eigen::VectorXf> y) {
    
    float RSS = (y - X * W).squaredNorm();
    float TSS = std::pow(((y - y.mean() * Eigen::VectorXf::Ones(y.rows())).lpNorm<2>()),2);

    return 1 - RSS/TSS;
    
}

int main() {

    const int N {100};
    const int P {2};
    const int batch_size {80};

    Eigen::Matrix<float, N, P+1> X;
    Eigen::Vector<float, N> y;
    std::string read_URL = "/Users/kailashanand/Documents/Developer/MC_OASIS/Task_Week4/amazing_data_0.csv";
    std::string write_URL = "/Users/kailashanand/Documents/Developer/MC_OASIS/Task_Week4/loss_history_0.csv";

    load_data_to(X, y, read_URL);
    // std::cout << "printing the matrices:\n";

    // std::cout << X << std::endl;
    // std::cout << y << std::endl;

    Eigen::Vector<float, P+1> W = Eigen::Vector<float, P+1>::Random() * 0.01;
    
    
    std::vector<float> loss_history = train_weights_with_LASSO_into(W, X, y, 0.01, 50000, 5000, 1e-2, false);
    std::cout << W << std::endl;

    std::cout << "R^2 value: " << find_R_squared_on(W, X, y) << "\n";

    write_vector_data_to(write_URL, loss_history);

    return 0;
}
