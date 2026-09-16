#pragma once

#include <iostream>
#include <fstream>

#include <vector>
#include <cmath>
#include <string>
#include <set>
#include <map>
#include <random>

#include "Eigen/Dense"
#include "Eigen/Core"

#include "load.h"

/*
An object that runs a GLM using the elastic net penalty on the l2 norm of the data loss in a linear regression setting.
Update: it now also has a TORRENT algorith to 

Future:
Will add an update to TORRENT that uses ideas from the glmnet paper to subselect both features and data-points.
*/
class GLMNET {

private:

    // Code to standardize X, and store the corresponding means and variances (return type) so that we can revert the standardization later to retrieve the correct weights
    Eigen::MatrixXd standardize_data(Eigen::Ref<Eigen::MatrixXd> X);


    // Used to reverse the standardization done on the data set to recover the correct weights

    // transformation_values should have dimension 2 x P: first row should be all the means, second row should be all the stds
    // weights should be a vector of dimension P+1
    void revert_transformation_on(Eigen::Ref<Eigen::MatrixXd> transformation_values);

    // The soft thresholding operator used to deal with the 3 cases for the value of sign(\beta_j)
    double soft_threshold(double z, double g);


    // this performs coordinate-by-coordinate cyclical coordinate descent using explicit for loops;
    void perform_CD_naive_on(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y, 
        int iters_till_stop, double tolerance, double reg, double alpha, bool quiet);

    // using a slightly better algorithm (as detailed in section 2.1 of the glmnet paper)
    void perform_CD_optimized1_on(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y,
        int iters_till_stop, double tolerance, double reg, double alpha, bool quiet);


    // Uses the vanilla covariance algorithm described in section 2.3 of the glmnet paper.
    void perform_CD_covariance_on(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y,
        int iters_till_stop, double tolerance, double reg, double alpha, bool quiet);




    // The following are methods to be used on adverserially corrupted response data.
    // They all use the TORRENT series of algorithms as presented in the RSLR paper.

    /*
    The Hard-thresholding operator, as defined in the TORRENT paper.
    1. residuals: These are residuals values used to judge/select the best k elements
    2. k: The number of (believed) clean examples
    3. index_set: The "Active set" of examples (index_set contains the corresponding indices)
    4. present: Keeps track of which indices were active at the end of each iteration.
    */
    double HT_into(Eigen::Ref<Eigen::VectorXd> residuals, int k, Eigen::Ref<Eigen::VectorXi> index_set, std::vector<int>& present);

    /*
    Fully Corrective TORRENT algorithm:
    At every update it performs a full on OLS update to the weights given the active set of examples
    1. X: The design/data matrix
    2. y: The responses (including the potentially corrupted ones)
    3. index_set: The active set of indices (this is calculated by the program itself)
    */
    void TORRENT_FC_update(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXi> index_set);

    /*
    Gradient Descent based TORRENT algorithm:
    At every update, it peforms a single gradient descent step on the weight vector.
    1. X: The design/data matrix
    2. y: The responses (including the potentially corrupted ones)
    3. index_set: The active set of indices (this is calculated by the program itself)
    4. gd_step: The learning rate of the gradient descent steps.
    */
    void TORRENT_GD_update(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y, Eigen::Ref<Eigen::VectorXi> index_set, 
        const double& gd_step);

    /*
    The Hybrid TORRENT update
    It switches between FC and GD updates depending on the stability of convergence in index_set.
    */
    void TORRENT_HYB_update(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y, 
        Eigen::Ref<Eigen::VectorXi> index_set, const double& gd_step, double delta, double index_set_delta);

public:
    int N;
    int P;

    // The weight vector that determines the GLM.
    Eigen::VectorXd W;

    // Initialzier
    /*
    1. n: The number of examples in the dataset you plan to load
    2. p: The number of predictors (excluding the bias) in the dataset you plan to load
    */
    GLMNET(int n, int p);

    
    /*
    Initializer to load the data while creating the GLM object.
    Note: Refer to the load_data_into function to see the expected format of X,y and the .csv file in URL.
    1. n: The number of examples in the dataset
    2. p: The number of predictors (excluding the bias term) in the dataset
    3. X: The (empty) matrix: nx(p+1) that is to be filled with the data stored in the URL
    4. y: The (empty) vector: (n,) that is to be filled with the corresponding 
    */
    GLMNET(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::string URL);

    /*
    data: The matrix (of pre-defined size) that you want to load the data into
    N: The number of examples in the datasat
    P: The number of predictors (excluding the bias)

    Convention that I am assuming:
    1. The data is already standardized
    2. The .csv file has no header, no SI.No column and all
    3. Each row is an example
    4. I will be adding 1 column of 1s (bias trick)
    5. First P columns are the predictors, the last column is response

    The sizes I expect:
    1. X: N x P+1
    2. y: N x 1
    */
    void load_data_into(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y, const std::string& URL);

    /*
    Trains the GLM using the covariance updates idea - detailed in Section 2.3 of the glmnet paper.
    (Refer to the load_data_function to know the expected sizes of X and y)
    1. X: The design matrix
    2. y: The response vector
    3. iters_till_stop: The upper limit on the number of iterations to perform if the weights never "converge" (convergence is defined by the tolerance you set)
    4. tolerance: The maximum L1 norm (error) you are willing you accept between the updated weights and old weights in each cycle 
    5. reg: The regularization strength, usually denoted by "lambda"
    6. alpha: The fraction of the total strength you assign the RIDGE component of the elastic net
    7. quiet: If this is false, then you will see console outputs on when it converged, etc. Setting quiet to true just mutes the function.

    Note: If your data isn't standardized, this will standardize it
    */
    void train_on(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y, 
        int iters_till_stop, double tolerance, double reg, double alpha, bool quiet);


    /*
    Uses TORRENT_HYB to find W based on X and y.
    Doesn't do any predictor selection.
    1. X: The design/data matrix
    2. The response vector
    3. iters_till_stop: The maximum number of iterations to perform in case the weights don't converge.
    4. tolerance: The maximum difference between two residual successive L2 norms before accepting convergence
    5. g: The fraction of examples believed to be corrupt
    6. gd_step: The learning rate used for a single gradient descent step
    7. delta: The maximum fractional of changes values between two successive index_sets before the programs considers updates to be unstable. 
        Set delta = 0 for a pure GD algorithm, delta=1 for a pure FC algorithm.
    8. quiet: If this is false, you will be outputs on the terminal about when it converged, etc. Setting quiet = true just mutes the function.

    Note: If your data isn't standardized, this will standardize it.
    */
    void train_with_RSLR_on(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y, 
        int iters_till_stop, double tolerance, double g, double gd_step, double delta, bool quiet);

    void predict_on(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> response);

};