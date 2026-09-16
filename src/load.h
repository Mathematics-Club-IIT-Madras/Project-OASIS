#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "Eigen/Dense"


/*
A function that converts a string_view into a double using the fast_float library.
1. sv: The string_view that needs to be converted into the double.
*/
double string_to_double(std::string_view sv);


/*
Loads data from the .csv file into an Eigen Matrix.
1. filename: The string that points to the location of the .csv file
2. X: An (N x P+1) matrix that contains the .csv file other than the y values. The initial size of X can be anything (empty is also OK)
3. y: The response vector in the .csv file
4. skip_header: true when the .csv file contains a header
*/
void csv_to_matrix(std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &y, bool skip_header);
