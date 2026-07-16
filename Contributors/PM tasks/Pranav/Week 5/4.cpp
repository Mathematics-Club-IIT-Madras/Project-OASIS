#include <iostream>
#include <vector>
#include <omp.h>
#define num 10

int main(){
    int n=100000000;

    std::vector<double> arr(n);
    for(int i=0;i<n;i++){
        arr[i]=1;
    }

    omp_set_num_threads(num);
    double total=0;
    double start=omp_get_wtime();

    #pragma omp parallel for reduction(+:total)
    for(int i=0;i<n;i++){
        total+=arr[i];
    }

    double end=omp_get_wtime();
    std::cout<<total<<" "<<end-start<<"\n";

    return 0;
}