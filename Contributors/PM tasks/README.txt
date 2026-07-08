Observations/Learnings:

1. For the data set generated with b_2 = 0.0001:
(using hyperparameters: learning_rate: 0.01, num_iters: 50000, period: 5000, on the data in: amazing_data.csv)

    a) OLS estimated b_0, b_1 quite accurately, but was ordered of magnitude off for b_2
    b) LASSO with reg: 3e-2, gave "worse" estimates for b_0, b_1, but estimates b_2 = 0.000347955, which
        is the correct order of magnitude
    c) With RIDGE on the other hand, I was unable to find a set of hyperparameters that got b_2 close to 
        the order of magnitude. It kept estimating it to be close to the OLS estimate itself (atleast in terms of the order of magnitude)
        This shows LASSOs advantage in being able to capture the correct order of magnitude (if not set to 0)

        Similarly, I generated synthetic data set: amazing_data_0.csv which b_2 set to 0. LASSO(reg=0.01) got b_2 estimate of
        -3.50458e-05! Pretty close to 0. It estimated the other coeffs to be: 0.520 and 2.951 - pretty accurate.

        Same story as before for RIDGE, couldn't get it to be any smaller than about order of 1e-2.

2. Pass by reference for Eigen objects is using Eigen::Ref<> and not the usual & sizing
3. We can use unary operators to perform custom functions over an entire matrix (i couldnt find a direct version of this in pytorch)


Areas of Improvement for my code itself:

1. Numerically unstable, if the data points become larger in magnitude then cooked. 
Need to add some normalization, some data processing step after parsing data into Eigen.

Why didn't I do it? Well, because if I normalize data and find estimates, then I have to perform some inverse
transform on the estimates for it to work properly on unknown test data. I don't know what thsoe transforms
are exactly.

2. Eigen datatype stuff : I kinda misused dynamic sizing, didn't put const keyword in some places
That would surely increase speed.

3. Optimizations: 
    a) I am recomputing X*W a few too many times, I'm using std::stof when 
        quicker alternatives are available (just to name a few)
    b) Data reading step requires 2 nested for loops. Can we do some parallel processing?
    c) Data reading also requires CSV and variables (N,P) to be in the perfect shape to work properly. 
        No proper error handling is there
    d) We can perform "better" GD algorithms (like mini-batch, Adam) to get faster convergence.

4. Put in class 

5. No Validation, Cross Validation (I couldn't figure out how to slice eigen matrices. I kept getting errors. I'll figure it out soon)