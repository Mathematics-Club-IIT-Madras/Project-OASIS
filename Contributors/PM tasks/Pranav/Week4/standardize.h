#include <Eigen/Dense>


void standardize_train(Eigen::MatrixXd &X,Eigen::VectorXd &means,Eigen::VectorXd &stddev);
void standardize_val(Eigen::MatrixXd &X,Eigen::VectorXd &means,Eigen::VectorXd &stddev);
void predict(Eigen::MatrixXd& X,Eigen::VectorXd& beta_hat,Eigen::VectorXd& Y_hat);
double find_mse(Eigen::VectorXd& Y,Eigen::VectorXd& Y_hat);
void centralise(Eigen::VectorXd &Y_train,Eigen::VectorXd &Y_val);
