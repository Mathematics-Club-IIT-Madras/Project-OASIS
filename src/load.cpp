#include <iostream>
#include <fstream>

#include <string>
#include <string_view>

#include <utility>
#include <vector>

#include "fast_float/fast_float.h"
#include "Eigen/Dense"

#include <system_error>
#include <stdexcept>




double string_to_double(std::string_view& sv) {
    double num;

    auto [ptr, err] = fast_float::from_chars(sv.data(), sv.data() + sv.size(), num);
    // ptr points to the start part of the string which is not numeric.

    if (err != std::errc{}) {

        throw std::runtime_error("Failed to parse float from token: '" + std::string(sv) + "'");
    }


    return num;
}


void csv_to_matrix(std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &y, bool skip_header) {
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;

    std::size_t rows = 0;
    std::size_t cols = 0;

    if (skip_header && getline(file, line))
        ; // NO OPERATION if 1st line is header.

    std::vector<double> buffer;
    buffer.reserve(500000);

    while (std::getline(file, line))
    {
        std::string_view l_view(line);

        std::size_t start = 0;
        int line_cols = 0;

        buffer.push_back(1);
        line_cols++;
        
        while (true)
        {   
            std::size_t end = l_view.find(',', start);
            if (end == std::string_view::npos)
            {
                std::string_view token = l_view.substr(start);
                buffer.push_back(string_to_double(token));
                line_cols++;
                break;
            }
            std::string_view token = l_view.substr(start, end - start);
            buffer.push_back(string_to_double(token));
            line_cols++;
            start = end + 1;
        }
        if (cols == 0)
        {
            cols = line_cols;
        }
        rows++;
    }

    size_t x_cols = cols - 1;

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> full_matrix(buffer.data(), rows, cols);

    X = full_matrix.leftCols(x_cols);
    y = full_matrix.rightCols(1);
}

