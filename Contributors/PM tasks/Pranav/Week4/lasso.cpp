#include "lasso.h"
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

void lasso_fit(Eigen::MatrixXd &X,Eigen::VectorXd &Y,double lamda,Eigen::VectorXd &beta_hat,int cycles,double diff){
    int p=X.cols();
    int n=X.rows();
    beta_hat= Eigen::VectorXd::Zero(p);

    for(int i=0;i<cycles;i++){
        double max_diff=0;
        for(int j=0;j<p;j++){

            Eigen::VectorXd rj=Y-X*beta_hat+X.col(j)*beta_hat(j);

            double zj=(X.col(j).array()*rj.array()).sum()/n;
            double new_beta_j;
            double old_beta_j=beta_hat(j);

            if(zj>lamda) new_beta_j=zj-lamda;
            else if(zj<-lamda) new_beta_j=zj+lamda;
            else new_beta_j=0;

            beta_hat[j]=new_beta_j;

            max_diff=std::max(max_diff,std::abs(new_beta_j-old_beta_j));
        }
        if(diff>max_diff){break;}
    }
}
