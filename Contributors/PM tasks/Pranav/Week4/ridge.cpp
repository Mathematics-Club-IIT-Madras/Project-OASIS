#include "ridge.h"
#include <Eigen/Dense>

void ridge_fit(Eigen::MatrixXd& X,Eigen::VectorXd& Y,double lamda,Eigen::VectorXd& beta_hat) {
    int p=X.cols();
    Eigen::MatrixXd XTX= X.transpose() * X;
    Eigen::MatrixXd XTY= X.transpose() * Y;
    Eigen::MatrixXd XTX_pro= XTX + lamda * Eigen::MatrixXd::Identity(p, p);
    beta_hat=XTX_pro.ldlt().solve(XTY);
}












