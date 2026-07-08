#include "load.h"
#include "ridge.h"
#include "lasso.h"
#include "standardize.h"
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

int main(){
    std::string filename="california_housing.csv";
    std::vector<std::vector<double>>rows;
    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
    file_to_rows(filename,rows);
    rows_to_matrix(X,Y,rows);

    int n=X.rows();
    int n_train=0.7*n;
    Eigen::MatrixXd X_train=X.topRows(n_train);
    Eigen::MatrixXd X_val=X.bottomRows(n-n_train);
    Eigen::VectorXd Y_train = Y.head(n_train);
    Eigen::VectorXd Y_val = Y.tail(n-n_train);

    Eigen::VectorXd means, stddev;
    standardize_train(X_train, means, stddev);
    standardize_val(X_val, means, stddev);
    centralise(Y_train,Y_val);

    int count=50;
    double min_loglamda=-2, max_loglamda=2;
    double temp=(max_loglamda-min_loglamda)/(count-1);

    std::ofstream ridge_out("output_ridgecoeff.csv");
    std::ofstream lasso_out("output_lassocoeff.csv");
    ridge_out<< "lamda";
    lasso_out<< "lamda";
    for(int i=1;i<=X.cols();i++){
        ridge_out<<",beta_"<<i;
        lasso_out<<",beta_"<<i;
    }
    ridge_out<<"\n";
    lasso_out<<"\n";

    for(int i=0;i<count;i++){
        double num=min_loglamda + i*temp;
        double lamda=std::pow(10,num);

        Eigen::VectorXd beta_hat_ridge, beta_hat_lasso;
        double cycles=1000,diff=1e-6;
        ridge_fit(X_train,Y_train,lamda,beta_hat_ridge);
        lasso_fit(X_train,Y_train,lamda,beta_hat_lasso,cycles,diff);  
        ridge_out<<lamda;
        lasso_out<<lamda;
        for(int i=1;i<=X.cols();i++){
            ridge_out<<","<<beta_hat_ridge(i-1);
            lasso_out<<","<<beta_hat_lasso(i-1);
        }
        ridge_out<<"\n";
        lasso_out<<"\n";
    }
}
