// ===========================================================
// CSV Dataset Loading and Performance Comparison
// Using std::stringstream, std::from_chars and fast_float
// Statistical analysis using Eigen3
// ===========================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <charconv>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include "fast_float/fast_float.h"

// -----------------------------------------------------------
// Method 1:
// Load CSV using std::stringstream and std::stod
// -----------------------------------------------------------

Eigen::VectorXd LoadCSV_strStream(const std::string& filename){
    
    std::ifstream file(filename);

    if (!file.is_open())
        {std::cout << "Could not open file." << std::endl;
         return Eigen::VectorXd();}
    
    std::string line;
    std::getline(file, line);

    int count = 0;

    while (std::getline(file, line))
    {
        count++;}
    
    file.close();

    Eigen::VectorXd values(count);
    file.open(filename);
    if(!file.is_open())
    {
        std::cout << "Could not open file." << std::endl;
        return Eigen::VectorXd(); }
    
    std::getline(file, line);
    int i = 0;
    while(std::getline(file, line))
    {
        std::stringstream ss(line);

        std::string timestamp;
        std::string value;

        std::getline(ss, timestamp, ',');

        std::getline(ss, value);

        values(i) = std::stod(value);

        i++; 
    }
    file.close();
    return values;
}

// -----------------------------------------------------------
// Method 2:
// Load CSV using std::string_view and std::from_chars
// -----------------------------------------------------------

Eigen::VectorXd LoadCSV_FromChars(const std::string& filename)
{

    std::ifstream file(filename);

    if (!file.is_open())
        {std::cout << "Could not open file." << std::endl;
         return Eigen::VectorXd();}
    
    std::string line;
    std::getline(file, line);

    int count = 0;

    while (std::getline(file, line))
    {
        count++;}
    
    file.close();

    Eigen::VectorXd values(count);
    file.open(filename);
    if(!file.is_open())
    {
        std::cout << "Could not open file." << std::endl;
        return Eigen::VectorXd(); }
    
    std::getline(file, line);
    int i = 0;
    while (std::getline(file, line))
{
        std::size_t comma = line.find(',');
       
        std::string_view value(
        line.data() + comma + 1,
        line.size() - comma - 1 );
    
        double number;
        
        auto result = std::from_chars(
        value.data(),
        value.data() + value.size(),
        number );

        if (result.ec == std::errc())     //std::errc() means no error. result contains information if the conversion worked
        {
            values(i) = number;
            i++;
        }
    }

    file.close();
    return values;
    
}

// -----------------------------------------------------------
// Method 3:
// Load CSV using fast_float library
// -----------------------------------------------------------

Eigen::VectorXd LoadCSV_FastFloat(const std::string& filename)
{

    std::ifstream file(filename);

    if (!file.is_open())
        {std::cout << "Could not open file." << std::endl;
         return Eigen::VectorXd();}
    
    std::string line;
    std::getline(file, line);

    int count = 0;

    while (std::getline(file, line))
    {
        count++;}
    
    file.close();

    Eigen::VectorXd values(count);
    file.open(filename);
    if(!file.is_open())
    {
        std::cout << "Could not open file." << std::endl;
        return Eigen::VectorXd(); }
    
    std::getline(file, line);
    int i = 0;
    while (std::getline(file, line))
{
        std::size_t comma = line.find(',');
       
        std::string_view value(
        line.data() + comma + 1,
        line.size() - comma - 1 );
    
        double number;
        
        auto result = fast_float::from_chars(
        value.data(),
        value.data() + value.size(),
        number );

        if (result.ec == std::errc())     //std::errc() means no error. result contains information if the conversion worked
        {
            values(i) = number;
            i++;
        }
    }

    file.close();
    return values;
    
}

// -----------------------------------------------------------
// Compute and display basic statistical parameters
// using Eigen library
// -----------------------------------------------------------

void PrintStatistics(const Eigen::VectorXd& values)
{
    double mean = values.mean();

    double variance =
        (values.array() - mean).square().sum() / values.size();

    double standardDeviation = std::sqrt(variance);

    std::cout << "Number of values : "
              << values.size()
              << std::endl;

    std::cout << "Sum : "
              << values.sum()
              << std::endl;

    std::cout << "Mean : "
              << mean
              << std::endl;

    std::cout << "Minimum : "
              << values.minCoeff()
              << std::endl;

    std::cout << "Maximum : "
              << values.maxCoeff()
              << std::endl;

    std::cout << "Variance : "
              << variance
              << std::endl;

    std::cout << "Standard Deviation : "
              << standardDeviation
              << std::endl;
}

// -----------------------------------------------------------
// Benchmark different CSV loading methods using
// the chrono library
// -----------------------------------------------------------

void Benchmark_Stream(const std::string& filename)
{
    auto start = std::chrono::high_resolution_clock::now();

    const int NUM_RUNS = 1000;

    for(int i = 0; i < NUM_RUNS; i++)
    {
        Eigen::VectorXd values = LoadCSV_strStream(filename);
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = end - start;

    double milliseconds =
        std::chrono::duration<double, std::milli>(duration).count();

    double average = milliseconds / NUM_RUNS;

    std::cout << std::endl;
    std::cout << "Benchmark (stringstream + stod)" << std::endl;
    std::cout << "Average Time : "
              << average
              << " ms"
              << std::endl;
}

void Benchmark_FromChars(const std::string& filename)
{
    auto start = std::chrono::high_resolution_clock::now();

    const int NUM_RUNS = 1000;

    for (int i = 0; i < NUM_RUNS; i++)
    {
        Eigen::VectorXd values = LoadCSV_FromChars(filename);
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = end - start;

    double milliseconds =
        std::chrono::duration<double, std::milli>(duration).count();

    double average = milliseconds / NUM_RUNS;

    std::cout << std::endl;
    std::cout << "Benchmark (string_view + from_chars)" << std::endl;
    std::cout << "Average Time : "
              << average
              << " ms"
              << std::endl;
}

void Benchmark_FastFloat(const std::string& filename)
{
    auto start = std::chrono::high_resolution_clock::now();

    const int NUM_RUNS = 1000;

    for (int i = 0; i < NUM_RUNS; i++)
    {
        Eigen::VectorXd values = LoadCSV_FastFloat(filename);
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = end - start;

    double milliseconds =
        std::chrono::duration<double, std::milli>(duration).count();

    double average = milliseconds / NUM_RUNS;

    std::cout << std::endl;
    std::cout << "Benchmark (fast_float)" << std::endl;
    std::cout << "Average Time : "
              << average
              << " ms"
              << std::endl;
}

// -----------------------------------------------------------
// Main Program
// 1. Load the dataset
// 2. Compute statistical parameters
// 3. Compare execution time of different loading methods
// -----------------------------------------------------------

int main()
{
    std::string filename = "art_daily_perfect_square_wave.csv";

    // Load once (using any method) to find the statistics
    Eigen::VectorXd values = LoadCSV_FastFloat(filename);

    // Statistics
    PrintStatistics(values);

    // Compare all three methods
    Benchmark_Stream(filename);

    Benchmark_FromChars(filename);

    Benchmark_FastFloat(filename);

    return 0;
}