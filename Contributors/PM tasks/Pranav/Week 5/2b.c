#include <stdio.h>
#include <omp.h>
#define num 10

int main(){
    int steps=10000000;
    double step_len=1.0/steps;
    double pi=0;
    omp_set_num_threads(num);
    double start=omp_get_wtime();
    #pragma omp parallel
    {
        int id=omp_get_thread_num();
        int nthreds=omp_get_num_threads();
        double sum=0;
        for(int i=id;i<steps;i+=nthreds){
            double x=(step_len)*(i+0.5);
            sum+=4/(1+x*x);
        }
        #pragma omp critical
        {
            pi+=sum*step_len;
        }
    }
    double end=omp_get_wtime();
    printf("%f %f\n",pi,end-start);
}