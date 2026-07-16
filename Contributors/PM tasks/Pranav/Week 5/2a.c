#include <stdio.h>
#include <omp.h>

int main(){
    int steps=10000000;
    double step_l=1.0/steps;
    double sum=0;
    double start=omp_get_wtime();
    for(int i=0;i<steps;i++){
        double x=(step_l)*(i+0.5);
        sum+=4/(1+x*x);
    }
    double end=omp_get_wtime();
    double pi=step_l*sum;
    printf("%f %f\n",pi,end-start);
}