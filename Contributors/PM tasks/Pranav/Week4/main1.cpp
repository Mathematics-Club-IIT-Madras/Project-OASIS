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

    std::ofstream out_ridge("mse_ridge.csv");
    std::ofstream out_lasso("mse_lasso.csv");
    out_ridge<< "lamda,train_mse,val_mse\n";
    out_lasso<< "lamda,train_mse,val_mse\n";

    for(int i=0;i<count;i++){
        double num=min_loglamda + i*temp;
        double lamda=std::pow(10,num);

        Eigen::VectorXd beta_hat_ridge,beta_hat_lasso;
        ridge_fit(X_train,Y_train,lamda,beta_hat_ridge);
        double cycles=1000,diff=1e-6;
        lasso_fit(X_train,Y_train,lamda,beta_hat_lasso,cycles,diff);

        Eigen::VectorXd Y_train_ridge_hat,Y_train_lasso_hat, Y_val_ridge_hat,Y_val_lasso_hat;
        predict(X_train,beta_hat_ridge,Y_train_ridge_hat);
        predict(X_val,beta_hat_ridge,Y_val_ridge_hat);
        predict(X_train,beta_hat_lasso,Y_train_lasso_hat);
        predict(X_val,beta_hat_lasso,Y_val_lasso_hat);

        double train_ridge_mse=find_mse(Y_train,Y_train_ridge_hat);
        double val_ridge_mse=find_mse(Y_val,Y_val_ridge_hat);
        double train_lasso_mse=find_mse(Y_train,Y_train_lasso_hat);
        double val_lasso_mse=find_mse(Y_val,Y_val_lasso_hat);

        out_ridge<<lamda<<","<<train_ridge_mse<<","<<val_ridge_mse<<"\n";
        out_lasso<<lamda<<","<<train_lasso_mse<<","<<val_lasso_mse<<"\n";
    }
}
