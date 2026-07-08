#include <cmath>
#include <Eigen/Dense>


void standardize_train(Eigen::MatrixXd &X,Eigen::VectorXd &means,Eigen::VectorXd &stddev){
    int n=X.rows(),p=X.cols();
    means.resize(p);
    stddev.resize(p);
    for(int i=0;i<p;i++){
        means(i)=X.col(i).mean();
        double sumsq=(X.col(i).array()-means(i)).square().sum();
        stddev(i)=std::sqrt(sumsq/n);
        X.col(i)=(X.col(i).array()-means(i))/stddev(i);
    }
}

void standardize_val(Eigen::MatrixXd &X,Eigen::VectorXd &means,Eigen::VectorXd &stddev){
    for(int i=0;i<X.cols();i++){
        X.col(i)=(X.col(i).array()-means(i))/stddev(i);
    }
}

void centralise(Eigen::VectorXd &Y_train,Eigen::VectorXd &Y_val){
    double mean=Y_train.mean();
    Y_train.array()-=mean;
    Y_val.array()-=mean;
}

void predict(Eigen::MatrixXd& X,Eigen::VectorXd& beta_hat,Eigen::VectorXd& Y_hat){
    Y_hat=X * beta_hat;
} 

double find_mse(Eigen::VectorXd& Y,Eigen::VectorXd& Y_hat){
    return (Y-Y_hat).array().square().mean();
}
