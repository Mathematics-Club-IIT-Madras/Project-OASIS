#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

int main(){
    ifstream file("data.csv");
    string line;
    getline(file, line); 
    vector<double>v;

    while(getline(file,line)){
        stringstream ss(line);
        string time,num;
        getline(ss,time,',');
        getline(ss,num,',');
        v.push_back(stod(num));
    }

    int l=v.size();

    VectorXd V(l);
    for(int i=0;i<l;i++){
        V(i)=v[i];
    }

    //Mean
    cout<<"Mean: "<<V.mean()<<"\n";

    //Variance
    double var=(V.array()-V.mean()).square().sum()/l;
    cout<<"Variance: "<<var<<"\n";

    //Median
    sort(v.begin(),v.end());
    double median;
    if(l%2) median=v[l/2];
    else median=(v[l/2]+v[l/2-1])/2.0;
    cout<<"Median: "<<median<<"\n";
}
