# Week 3

## Objective
Compare different methods for loading CSV files efficiently. ( Using Eigen )

## Files
- optimization.cpp
- art_daily_perfect_square_wave.csv

## Methods Compared
1. std::stringstream
2. std::from_chars
3. fast_float

## Benchmarking
Used the `<chrono>` library to measure execution time of each implementation.

## Results
- Compared average execution time over multiple runs.
- Observed significant speed improvement using `std::from_chars` compared to `std::stringstream`.
