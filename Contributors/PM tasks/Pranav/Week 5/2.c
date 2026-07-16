#include <stdio.h>
#include <omp.h>
#define num 10

int main(){
    int steps=10000000;
    double step_len=1.0/steps;
    int nthreads;
    double sum[num][8]={0};
    omp_set_num_threads(num);
    double start=omp_get_wtime();
    #pragma omp parallel
    {
        int id=omp_get_thread_num();
        int nthreds=omp_get_num_threads();
        if(id==0) nthreads=nthreds;
        for(int i=id;i<steps;i+=nthreds){
            double x=(step_len)*(i+0.5);
            sum[id][0]+=4/(1+x*x);
        }
    }
    double pi=0;
    for(int id=0;id<nthreads;id++) pi+=sum[id][0]*step_len;
    double end=omp_get_wtime();
    printf("%f %f\n",pi,end-start);
}