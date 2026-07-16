#include <iostream>
#include <vector>
#include <omp.h>
#define num 10

int main(){
    int n=500;
    std::vector<std::vector<double>>A(n,std::vector<double>(n));
    std::vector<std::vector<double>>B(n,std::vector<double>(n));
    std::vector<std::vector<double>>C(n,std::vector<double>(n,0));

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            A[i][j]=i+1;
            B[i][j]=j-1;
        }
    }

    omp_set_num_threads(num);
    double start=omp_get_wtime();

    #pragma omp parallel for
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            double sum=0;
            for(int k=0;k<n;k++){
                sum+=A[i][k]*B[k][j];
            }
            C[i][j]=sum;
        }
    }

    double end=omp_get_wtime();
    std::cout<<end-start<<"\n";

    return 0;
}