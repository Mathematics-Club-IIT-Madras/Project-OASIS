#include <Eigen/Dense>

void lasso_fit(Eigen::MatrixXd& X,Eigen::VectorXd& Y,double lamda,Eigen::VectorXd &beta_hat,int cycles,double diff);
