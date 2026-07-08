#include <string>
#include <string_view>
#include <vector>
#include <Eigen/Dense>

double string_to_double(std::string_view sv);
void file_to_rows(std::string& filename,std::vector<std::vector<double>>&rows);
void rows_to_matrix(Eigen::MatrixXd &X,Eigen::VectorXd &Y,const std::vector<std::vector<double>>&rows);
